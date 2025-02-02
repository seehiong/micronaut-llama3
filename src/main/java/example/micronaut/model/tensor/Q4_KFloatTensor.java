package example.micronaut.model.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import example.micronaut.gguf.Float16;
import example.micronaut.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Q4_KFloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public Q4_KFloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q4_K;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        int blockIndex = index / GGMLType.Q4_K.getBlockSize();
        int blockOffset = blockIndex * GGMLType.Q4_K.getTypeSize();
        float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        float zeroPoint = Float.float16ToFloat(readShort(memorySegment, blockOffset + Float16.BYTES));

        byte quant;
        int modIndex = index % GGMLType.Q4_K.getBlockSize();
        if (modIndex < GGMLType.Q4_K.getBlockSize() / 2) {
            quant = (byte) (readByte(memorySegment, blockOffset + 2 * Float16.BYTES + modIndex) & 0x0F);
        } else {
            quant = (byte) ((readByte(memorySegment,
                    blockOffset + 2 * Float16.BYTES + modIndex - GGMLType.Q4_K.getBlockSize() / 2) >>> 4) & 0x0F);
        }
        quant -= 8;
        return quant * scale + zeroPoint;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && size > F_SPECIES.length()) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return super.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(Q4_KFloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset,
            int size) {
        float result = 0f;
        int j = 0;

        final int blockSize = GGMLType.Q4_K.getBlockSize();
        final int typeSize = GGMLType.Q4_K.getTypeSize();

        int alignmentMask = blockSize - 1;
        int alignmentBound = Math.min(size, -thisOffset & alignmentMask);
        if (alignmentBound > 0) {
            result += super.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        FloatVector accumulator = FloatVector.zero(F_SPECIES);
        int blockOffset = (thisOffset + j) / blockSize * typeSize;
        int upperBound = size / (blockSize * blockSize);

        for (; j < upperBound; j += blockSize, blockOffset += typeSize) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            float zeroPoint = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset + Float16.BYTES));
            FloatVector wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            FloatVector wZeroPoint = FloatVector.broadcast(F_SPECIES, zeroPoint);

            ByteVector wBytes = ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_128,
                    thiz.memorySegment,
                    blockOffset + 2 * Float16.BYTES,
                    ByteOrder.LITTLE_ENDIAN
            );

            ByteVector loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            ByteVector hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);

            processVectorDot(that, thatOffset, j, loBytes, hiBytes, wScale, wZeroPoint, accumulator);
        }

        result += accumulator.reduceLanes(VectorOperators.ADD);

        if (j < size) {
            result += super.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private void processVectorDot(
            ArrayFloatTensor that,
            int thatOffset,
            int j,
            ByteVector loBytes,
            ByteVector hiBytes,
            FloatVector wScale,
            FloatVector wZeroPoint,
            FloatVector accumulator
    ) {
        float[] loFloats = new float[F_SPECIES.length()];
        float[] hiFloats = new float[F_SPECIES.length()];

        for (int i = 0; i < F_SPECIES.length(); i++) {
            loFloats[i] = loBytes.lane(i); // Convert byte to float
            hiFloats[i] = hiBytes.lane(i);
        }

        FloatVector adjustedLo = FloatVector.fromArray(F_SPECIES, loFloats, 0)
                .mul(wScale)
                .add(wZeroPoint);

        FloatVector adjustedHi = FloatVector.fromArray(F_SPECIES, hiFloats, 0)
                .mul(wScale)
                .add(wZeroPoint);

        var sum0 = that.getFloatVector(F_SPECIES, thatOffset + j + 0 * F_SPECIES.length()).mul(adjustedLo);
        var sum1 = that.getFloatVector(F_SPECIES, thatOffset + j + 1 * F_SPECIES.length()).mul(adjustedHi);

        accumulator.add(sum0).add(sum1);
    }
}
