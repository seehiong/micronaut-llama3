package example.micronaut.model.tensor;

import sun.misc.Unsafe;

import java.lang.foreign.MemorySegment;
import java.lang.reflect.Field;

import example.micronaut.gguf.GGMLType;
import example.micronaut.utils.Parallel;
import example.micronaut.utils.TransformerUtils;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorShape;
import jdk.incubator.vector.VectorSpecies;

/**
 * Over-simplified, shapeless, float tensor.
 * <p>
 * Not a strict tensor, but rather just a sequence of floats, not required to be
 * backed by memory e.g. can represent a sequence of quantized floats.
 */
public abstract class FloatTensor {

    public static final int VECTOR_BIT_SIZE = Integer.getInteger("llama.VectorBitSize",
            VectorShape.preferredShape().vectorBitSize());
    public static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    // static final ValueLayout.OfFloat JAVA_FLOAT_LE =
    // ValueLayout.JAVA_FLOAT.withOrder(ByteOrder.LITTLE_ENDIAN);
    // static final ValueLayout.OfShort JAVA_SHORT_LE =
    // ValueLayout.JAVA_SHORT.withOrder(ByteOrder.LITTLE_ENDIAN);
    // The use of Unsafe in this file is a temporary workaround to support
    // native-image.
    public static final Unsafe UNSAFE;

    static {
        try {
            Field f = Unsafe.class.getDeclaredField("theUnsafe");
            f.setAccessible(true);
            UNSAFE = (Unsafe) f.get(null);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException(e);
        }
    }

    protected short readShort(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead but it's slower
        // return memorySegment.get(ValueLayout.JAVA_SHORT, offset); 
        return UNSAFE.getShort(memorySegment.address() + offset);
    }

    protected byte readByte(MemorySegment memorySegment, long offset) {
        // The MemorySegment.get* methods should be used instead but it's slower
        // return memorySegment.get(ValueLayout.JAVA_BYTE, offset);
        return UNSAFE.getByte(memorySegment.address() + offset);
    }

    // Preferred vector size for the fast multiplication routines.
    // (Apple Silicon) NEON only supports up-to 128bit vectors.
    public static final VectorSpecies<Float> F_SPECIES;
    public static final VectorSpecies<Integer> I_SPECIES;
    public static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF = VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    public abstract int size();

    public abstract float getFloat(int index);

    public abstract void setFloat(int index, float value);

    public abstract FloatVector getFloatVector(VectorSpecies<Float> species, int offset);

    public abstract GGMLType type();

    protected float scalarDot(FloatTensor thiz, int thisOffset, FloatTensor that, int thatOffset, int size) {
        float result = 0f;
        for (int j = 0; j < size; j++) {
            result += thiz.getFloat(thisOffset + j) * that.getFloat(thatOffset + j);
        }
        return result;
    }

    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return scalarDot(this, thisOffset, that, thatOffset, size);
    }

    public void matmul(FloatTensor that, FloatTensor out, int dim0, int dim1) {
        Parallel.parallelFor(0, dim0, i -> out.setFloat(i, dot(i * dim1, that, 0, dim1)));
    }

    public void matmul(int context, FloatTensor[] that, FloatTensor[] out, int dim0, int dim1) {
        if (that.length != out.length) {
            throw new IllegalArgumentException(String.format("that.len=%d, out.len=%d", that.length, out.length));
        }
        Parallel.parallelForLong(0, dim0 * context, ti -> {
            int idxArr = (int) (ti / dim0);
            int i = (int) (ti % dim0);
            out[idxArr].setFloat(i, dot(i * dim1, that[idxArr], 0, dim1));
        });
    }

    @FunctionalInterface
    public interface AggregateFunction {

        float apply(float acc, float value);
    }

    public float reduce(int thisOffset, int size, float seed, AggregateFunction reduce) {
        float result = seed;
        for (int i = 0; i < size; ++i) {
            result = reduce.apply(result, getFloat(thisOffset + i));
        }
        return result;
    }

    public float sum(int thisOffset, int size) {
        return reduce(thisOffset, size, 0f, Float::sum);
    }

    public float max(int thisOffset, int size) {
        return reduce(thisOffset, size, Float.NEGATIVE_INFINITY, Float::max);
    }

    public void copyTo(int thisOffset, FloatTensor that, int thatOffset, int size) {
        that.mapWithIndexInPlace(thatOffset, size, (value, index) -> this.getFloat(index - thatOffset + thisOffset));
    }

    public int argmax(int thisOffset, int size) {
        assert size > 0;
        int maxIndex = thisOffset;
        float maxValue = this.getFloat(maxIndex);
        int endIndex = thisOffset + size;
        for (int i = thisOffset; i < endIndex; ++i) {
            float f = this.getFloat(i);
            if (f > maxValue) {
                maxValue = f;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public int argmax() {
        return argmax(0, size());
    }

    @FunctionalInterface
    public interface MapFunction {

        float apply(float value);
    }

    @FunctionalInterface
    public interface MapWithIndexFunction {

        float apply(float value, int index);
    }

    public FloatTensor mapInPlace(int thisOffset, int size, MapFunction mapFunction) {
        // Pre-calculate end boundary
        final int endIndex = thisOffset + size;

        // Process in chunks to improve cache utilization
        final int CHUNK_SIZE = 1024;
        for (int chunk = thisOffset; chunk < endIndex; chunk += CHUNK_SIZE) {
            int chunkEnd = Math.min(chunk + CHUNK_SIZE, endIndex);
            for (int i = chunk; i < chunkEnd; i++) {
                setFloat(i, mapFunction.apply(getFloat(i)));
            }
        }
        return this;
    }

    public FloatTensor mapInPlace(MapFunction mapFunction) {
        return mapInPlace(0, size(), mapFunction);
    }

    public FloatTensor mapWithIndexInPlace(int thisOffset, int size,
            FloatTensor.MapWithIndexFunction mapWithIndexFunction) {
        int endOffset = thisOffset + size;
        for (int i = thisOffset; i < endOffset; ++i) {
            setFloat(i, mapWithIndexFunction.apply(getFloat(i), i));
        }
        return this;
    }

    public FloatTensor addInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size,
                (value, index) -> value + that.getFloat(index - thisOffset + thatOffset));
    }

    public FloatTensor addInPlace(FloatTensor that) {
        return addInPlace(0, that, 0, size());
    }

    public FloatTensor multiplyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size) {
        return mapWithIndexInPlace(thisOffset, size,
                (value, index) -> value * that.getFloat(index - thisOffset + thatOffset));
    }

    public FloatTensor multiplyInPlace(FloatTensor that) {
        return multiplyInPlace(0, that, 0, size());
    }

    public FloatTensor divideInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, f -> f / value);
    }

    public FloatTensor fillInPlace(int thisOffset, int size, float value) {
        return mapInPlace(thisOffset, size, unused -> value);
    }

    public FloatTensor softmaxInPlace(int thisOffset, int size) {
        // find max value (for numerical stability)
        float maxVal = max(thisOffset, size);
        // exp and sum
        mapInPlace(thisOffset, size, f -> (float) TransformerUtils.fastExp(f - maxVal));
        float sum = sum(thisOffset, size);
        // normalize
        return divideInPlace(thisOffset, size, sum);
    }

    public FloatTensor saxpyInPlace(int thisOffset, FloatTensor that, int thatOffset, int size, float a) {
        // this[thatOffset ... thatOffset + size) = 
        // a * that[thatOffset ... thatOffset + size) + this[thisOffset ... thisOffset + size)
        for (int i = 0; i < size; ++i) {
            setFloat(thisOffset + i, a * that.getFloat(thatOffset + i) + this.getFloat(thisOffset + i));
        }
        return this;
    }
}
