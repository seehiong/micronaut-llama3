package example.micronaut.model.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import example.micronaut.gguf.Float16;
import example.micronaut.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Q4_0FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_0;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_0.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_0.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        byte quant;
        int modIndex = index % GGMLType.Q4_0.getBlockSize();
        if (modIndex < GGMLType.Q4_0.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment,
                    blockOffset + Float16.BYTES + modIndex - GGMLType.Q4_0.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && size > F_SPECIES.length()) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset,
            int size) {
        float result = 0f;
        int j = 0;

        final int blockSize = GGMLType.Q4_0.getBlockSize();
        final int typeSize = GGMLType.Q4_0.getTypeSize();

        int alignmentMask = blockSize - 1;
        int alignmentBound = Math.min(size, -thisOffset & alignmentMask);
        if (alignmentBound > 0) {
            result += FloatTensor.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector accumulator = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / blockSize * typeSize;
        int upperBound = size / (blockSize * blockSize);

        // Vectorized main processing loop
        for (; j < upperBound; j += blockSize, blockOffset += typeSize) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            FloatVector wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);

            ByteVector wBytes = ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_128,
                    thiz.memorySegment,
                    blockOffset + GGMLType.FLOAT16_BYTES,
                    ByteOrder.LITTLE_ENDIAN
            );

            ByteVector loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            ByteVector hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            // Unified processing for different vector sizes
            processVectorDot(that, thatOffset, j, loBytes, hiBytes, wScale, accumulator);
        }

        // Reduce accumulated vector to scalar
        result += accumulator.reduceLanes(VectorOperators.ADD);

        // Handle remaining elements
        if (j < size) {
            result += FloatTensor.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    // Extract vector dot processing to a separate method for clarity and potential optimization
    private void processVectorDot(
            ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector wScale,
            FloatVector accumulator
    ) {
        switch (F_SPECIES.vectorBitSize()) {
            case 512 ->
                process512BitVector(that, thatOffset, j, loBytes, hiBytes, wScale, accumulator);
            case 256 ->
                process256BitVector(that, thatOffset, j, loBytes, hiBytes, wScale, accumulator);
            case 128 ->
                process128BitVector(that, thatOffset, j, loBytes, hiBytes, wScale, accumulator);
            default ->
                throw new UnsupportedOperationException(F_SPECIES.toString());
        }
    }

    private void process512BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector wScale,
            FloatVector accumulator) {
        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 0));
        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 0));
        sum0.add(sum2).fma(wScale, accumulator);
    }

    private void process256BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector wScale,
            FloatVector accumulator) {
        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 0));
        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length())
                .mul(loBytes.castShape(F_SPECIES, 1));
        var sum2 = that.getFloatVector(F_SPECIES, thatOffset + j + 2 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 0));
        var sum3 = that.getFloatVector(F_SPECIES, thatOffset + j + 3 * F_SPECIES.length())
                .mul(hiBytes.castShape(F_SPECIES, 1));
        sum0.add(sum1).add(sum2).add(sum3).fma(wScale, accumulator);
    }

    private void process128BitVector(ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector wScale,
            FloatVector accumulator) {
        // This loop cannot be unrolled, why?
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
            sum0.add(sum1).add(sum2).add(sum3).fma(wScale, accumulator);
        }
    }
}
