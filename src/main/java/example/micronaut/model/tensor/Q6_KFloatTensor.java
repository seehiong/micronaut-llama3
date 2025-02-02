package example.micronaut.model.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import example.micronaut.gguf.Float16;
import example.micronaut.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Q6_KFloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q6_KFloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void setFloat(int index, float value) {
        throw new UnsupportedOperationException("setFloat");
    }

    @Override
    public FloatVector getFloatVector(VectorSpecies<Float> species, int index) {
        throw new UnsupportedOperationException("getFloatVector");
    }

    @Override
    public GGMLType type() {
        return GGMLType.Q6_K;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;

        int blockIndex = index / GGMLType.Q6_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q6_K.getTypeSize();

        // Read scale
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));

        // Extract 6-bit quantized value
        int modIndex = index % GGMLType.Q6_K.getBlockSize();
        int byteIndex = modIndex * 6 / 8; // Position inside the packed bytes
        int bitOffset = (modIndex * 6) % 8; // Bit offset inside the byte

        byte packedByte = readByte(memorySegment, blockOffset + Float16.BYTES + byteIndex);
        int quantizedValue = (packedByte >>> bitOffset) & 0x3F; // Extract 6-bit value

        // Convert to signed (-32 to 31)
        if (quantizedValue >= 32) {
            quantizedValue -= 64; // Sign extension
        }

        return quantizedValue * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && size > F_SPECIES.length()) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return super.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(Q6_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        final int blockSize = GGMLType.Q6_K.getBlockSize();
        final int typeSize = GGMLType.Q6_K.getTypeSize();

        // Handle alignment
        int alignmentMask = blockSize - 1;
        int alignmentOffset = thisOffset & alignmentMask;
        int alignmentBound = alignmentOffset == 0 ? 0 : blockSize - alignmentOffset;
        if (alignmentBound > 0) {
            result += super.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector accumulator = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / blockSize * typeSize;
        int upperBound = size - (size % blockSize);

        for (; j < upperBound; j += blockSize, blockOffset += typeSize) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            FloatVector wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);

            ByteVector packedBytes = ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_128,
                    thiz.memorySegment,
                    blockOffset + Float16.BYTES,
                    ByteOrder.LITTLE_ENDIAN
            );

            ByteVector loBytes = extract6BitValues(packedBytes, 0);
            ByteVector hiBytes = extract6BitValues(packedBytes, 1);
            processVectorDot(that, thatOffset, j, loBytes, hiBytes, wScale, accumulator);
        }

        result += accumulator.reduceLanes(VectorOperators.ADD);

        // Handle remaining elements
        if (j < size) {
            result += super.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private ByteVector extract6BitValues(ByteVector packed, int index) {
        // Shift left to align bits
        int shiftLeftAmount = 8 - (index + 1) * 6;
        ByteVector shiftedLeft = packed.lanewise(VectorOperators.LSHL, shiftLeftAmount);

        // Shift right to bring desired bits to the least significant bits
        int shiftRightAmount = 8 - 6;
        ByteVector shiftedRight = shiftedLeft.lanewise(VectorOperators.LSHR, shiftRightAmount);

        // Mask to get the lowest 6 bits
        ByteVector masked = shiftedRight.and((byte) 0x3F);

        // Sign extension
        ByteVector signExtended = masked.sub((byte) 32);

        return signExtended;
    }

    private void processVectorDot(
            ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector scale, // Change from float to FloatVector
            FloatVector accumulator
    ) {
        switch (F_SPECIES.vectorBitSize()) {
            case 512 ->
                process512BitVector(that, thatOffset, j, loBytes, hiBytes, scale, accumulator);
            case 256 ->
                process256BitVector(that, thatOffset, j, loBytes, hiBytes, scale, accumulator);
            case 128 ->
                process128BitVector(that, thatOffset, j, loBytes, hiBytes, scale, accumulator);
            default ->
                throw new UnsupportedOperationException(F_SPECIES.toString());
        }
    }

    private void process512BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector scale,
            FloatVector accumulator) {
        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 0));
        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 0));
        accumulator.add(sum0.add(sum2).mul(scale));
    }

    private void process256BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector scale,
            FloatVector accumulator) {
        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 0));
        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 1));
        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 0));
        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 1));
        accumulator.add(sum0.add(sum1).add(sum2).add(sum3).mul(scale));
    }

    private void process128BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector scale,
            FloatVector accumulator) {
        for (int i = 0; i < 2; ++i) {
            var tmp = i == 0 ? loBytes : hiBytes;
            var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 0) * F_SPECIES.length())
                    .mul(tmp.castShape(F_SPECIES, 0));
            var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                    .mul(tmp.castShape(F_SPECIES, 1));
            var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                    .mul(tmp.castShape(F_SPECIES, 2));
            var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                    .mul(tmp.castShape(F_SPECIES, 3));
            accumulator = accumulator.add(sum0.add(sum1).add(sum2).add(sum3).mul(scale));
        }
    }
}
