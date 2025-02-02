package example.micronaut.model.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import example.micronaut.gguf.Float16;
import example.micronaut.gguf.GGMLType;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class Q8_0FloatTensor extends FloatTensor {

    private final int size;
    private final MemorySegment memorySegment;
    private static final int BLOCK_SIZE;
    private static final int TYPE_SIZE;
    private static final int BLOCK_SIZE_MASK;

    static {
        BLOCK_SIZE = GGMLType.Q8_0.getBlockSize();
        TYPE_SIZE = GGMLType.Q8_0.getTypeSize();
        BLOCK_SIZE_MASK = BLOCK_SIZE - 1;
    }

    public Q8_0FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.Q8_0;
    }

    @Override
    public float getFloat(int index) {
        //assert 0 <= index && index < size;
        final int blockIndex = index >>> Integer.numberOfTrailingZeros(BLOCK_SIZE);
        final int withinBlockIndex = index & BLOCK_SIZE_MASK;
        final int blockOffset = blockIndex * TYPE_SIZE;

        final byte quant = readByte(memorySegment, blockOffset + Float16.BYTES + withinBlockIndex);
        final float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));
        return quant * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return super.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(Q8_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Handle alignment
        final int alignmentBound = Math.min(size, (-thisOffset) & BLOCK_SIZE_MASK);
        if (alignmentBound > 0) {
            result += super.scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }

        // Main vector processing loop
        FloatVector accumulator = FloatVector.zero(F_SPECIES);
        final int upperBound = size & ~BLOCK_SIZE_MASK;

        for (; j < upperBound; j += BLOCK_SIZE) {
            final long blockOffset = ((thisOffset + j) / BLOCK_SIZE) * TYPE_SIZE;
            final float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            int baseOffset = thatOffset + j;

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    accumulator = that.getFloatVector(F_SPECIES, baseOffset).mul(wBytes.castShape(F_SPECIES, 0))
                            .add(that.getFloatVector(F_SPECIES, baseOffset + F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1)))
                            .fma(wScale, accumulator);
                }
                case 256 -> {
                    var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_256, thiz.memorySegment, blockOffset + Float16.BYTES, ByteOrder.LITTLE_ENDIAN);
                    accumulator = that.getFloatVector(F_SPECIES, baseOffset).mul(wBytes.castShape(F_SPECIES, 0))
                            .add(that.getFloatVector(F_SPECIES, baseOffset + F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1)))
                            .add(that.getFloatVector(F_SPECIES, baseOffset + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2)))
                            .add(that.getFloatVector(F_SPECIES, baseOffset + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3)))
                            .fma(wScale, accumulator);
                }
                case 128 -> {
                    // Process two 128-bit blocks sequentially
                    for (int i = 0; i < 2; i++) {
                        var wBytes = ByteVector.fromMemorySegment(ByteVector.SPECIES_128, thiz.memorySegment,
                                blockOffset + Float16.BYTES + i * ByteVector.SPECIES_128.vectorByteSize(), ByteOrder.LITTLE_ENDIAN);
                        baseOffset += i * 16;
                        accumulator = that.getFloatVector(F_SPECIES, baseOffset).mul(wBytes.castShape(F_SPECIES, 0))
                                .add(that.getFloatVector(F_SPECIES, baseOffset + F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 1)))
                                .add(that.getFloatVector(F_SPECIES, baseOffset + 2 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 2)))
                                .add(that.getFloatVector(F_SPECIES, baseOffset + 3 * F_SPECIES.length()).mul(wBytes.castShape(F_SPECIES, 3)))
                                .fma(wScale, accumulator);
                    }
                }
                default ->
                    throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }

        result += accumulator.reduceLanes(VectorOperators.ADD);

        // Handle remaining elements
        if (j < size) {
            result += super.scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

}
