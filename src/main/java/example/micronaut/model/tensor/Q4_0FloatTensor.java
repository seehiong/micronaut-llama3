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

    private final int size;
    private final MemorySegment memorySegment;
    private static final int BLOCK_SIZE;
    private static final int TYPE_SIZE;
    private static final int HALF_BLOCK_SIZE;
    private static final int BLOCK_SIZE_MASK; // For faster modulo operations
    private final ByteVector MASK_LOW;  // Cached mask vectors
    private final ByteVector OFFSET_8;
    private static final int VECTOR_ALIGNMENT = 32; // Typical CPU cache line size
    private final ByteVector BLEND_MASK; // For combining high and low nibbles
    private static final int BLOCK_SIZE_SQUARED;

    static {
        BLOCK_SIZE = GGMLType.Q4_0.getBlockSize();
        TYPE_SIZE = GGMLType.Q4_0.getTypeSize();
        HALF_BLOCK_SIZE = BLOCK_SIZE / 2;
        BLOCK_SIZE_MASK = BLOCK_SIZE - 1;
        BLOCK_SIZE_SQUARED = BLOCK_SIZE * BLOCK_SIZE;
    }

    public Q4_0FloatTensor(int size, MemorySegment memorySegment) {
        this.size = size;
        this.memorySegment = memorySegment;
        // Pre-compute commonly used vectors
        this.MASK_LOW = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 0x0F);
        this.OFFSET_8 = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 8);
        this.BLEND_MASK = ByteVector.broadcast(ByteVector.SPECIES_128, (byte) 0xF0);  // For high nibble
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
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index out of bounds: " + index);
        }
        final int blockIndex = index >>> Integer.numberOfTrailingZeros(BLOCK_SIZE);
        final long blockOffset = blockIndex * TYPE_SIZE;
        final float scale = Float.float16ToFloat(readShort(memorySegment, blockOffset));

        final int modIndex = index & BLOCK_SIZE_MASK;
        final long byteOffset = blockOffset + Float16.BYTES + (modIndex & ~HALF_BLOCK_SIZE);
        final byte quantByte = readByte(memorySegment, byteOffset);

        // Use bitwise masking instead of if-else
        final byte quant = (byte) ((quantByte >>> ((modIndex & HALF_BLOCK_SIZE) >>> 2)) & 0x0F);
        return (quant - 8) * scale;
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (FloatTensor.USE_VECTOR_API && size > F_SPECIES.length()) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return super.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(Q4_0FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Optimize alignment calculation
        final int alignmentOffset = (-thisOffset) & (VECTOR_ALIGNMENT - 1);
        if (alignmentOffset > 0) {
            result = scalarDot(this, thisOffset, that, thatOffset, alignmentOffset);
            j += alignmentOffset;
        }

        FloatVector accumulator = FloatVector.zero(F_SPECIES);
        int blockOffset = ((thisOffset + j) / BLOCK_SIZE) * TYPE_SIZE;
        int upperBound = size / BLOCK_SIZE_SQUARED;

        // Vectorized main processing loop with optimized memory access
        for (; j < upperBound; j += BLOCK_SIZE, blockOffset += TYPE_SIZE) {
            float wScaleValue = Float.float16ToFloat(readShort(thiz.memorySegment, blockOffset));
            FloatVector wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);

            ByteVector wBytes = ByteVector.fromMemorySegment(
                    ByteVector.SPECIES_128,
                    thiz.memorySegment,
                    blockOffset + Float16.BYTES,
                    ByteOrder.LITTLE_ENDIAN
            );

            ByteVector loBytes = wBytes.and(MASK_LOW).sub(OFFSET_8);
            ByteVector hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub(OFFSET_8);
            ByteVector combined = loBytes.bitwiseBlend(hiBytes.lanewise(VectorOperators.LSHL, 4), BLEND_MASK);

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum = that.getFloatVector(F_SPECIES, thatOffset + j).mul(combined.castShape(F_SPECIES, 0));
                    sum.fma(wScale, accumulator);
                }
                case 256 -> {
                    var sum = FloatVector.zero(F_SPECIES);
                    var vec0 = that.getFloatVector(F_SPECIES, thatOffset + j);
                    var vec1 = that.getFloatVector(F_SPECIES, thatOffset + j + F_SPECIES.length());
                    sum = sum.add(vec0.mul(combined.castShape(F_SPECIES, 0)));
                    sum = sum.add(vec1.mul(combined.castShape(F_SPECIES, 1)));
                    sum.fma(wScale, accumulator);
                }
                case 128 -> {
                    final int baseOffset = thatOffset + j;
                    var sum0 = that.getFloatVector(F_SPECIES, baseOffset)
                            .mul(combined.castShape(F_SPECIES, 0));
                    var sum1 = that.getFloatVector(F_SPECIES, baseOffset + F_SPECIES.length())
                            .mul(combined.castShape(F_SPECIES, 1));
                    var sum2 = that.getFloatVector(F_SPECIES, baseOffset + (2 * F_SPECIES.length()))
                            .mul(combined.castShape(F_SPECIES, 2));
                    var sum3 = that.getFloatVector(F_SPECIES, baseOffset + (3 * F_SPECIES.length()))
                            .mul(combined.castShape(F_SPECIES, 3));
                    sum0.add(sum1).add(sum2).add(sum3).fma(wScale, accumulator);
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
