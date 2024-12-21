package example.micronaut.model.tensor;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;

import example.micronaut.gguf.GGMLType;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.ShortVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class F16FloatTensor extends FloatTensor {

    final int size;
    final MemorySegment memorySegment;

    public F16FloatTensor(int size, MemorySegment memorySegment) {
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
        return GGMLType.F16;
    }

    @Override
    public float getFloat(int index) {
        assert 0 <= index && index < size;
        return Float.float16ToFloat(readShort(memorySegment, index * GGMLType.FLOAT16_BYTES));
    }

    @Override
    public float dot(int thisOffset, FloatTensor that, int thatOffset, int size) {
        if (USE_VECTOR_API) {
            return vectorDot(this, thisOffset, (ArrayFloatTensor) that, thatOffset, size);
        } else {
            return FloatTensor.scalarDot(this, thisOffset, that, thatOffset, size);
        }
    }

    private float vectorDot(F16FloatTensor thiz, int thisOffset, ArrayFloatTensor that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        int upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = that.getFloatVector(F_SPECIES, thatOffset + i);
            ShortVector bits16 = ShortVector.fromMemorySegment(S_SPECIES_HALF, thiz.memorySegment,
                    (thisOffset + i) * (long) GGMLType.FLOAT16_BYTES, ByteOrder.LITTLE_ENDIAN);

            var bits32 = bits16.castShape(I_SPECIES, 0).reinterpretAsInts(); // (int) bits16
            // Does not support infinities nor NaNs, preserves sign, emulate DAZ
            // (denormals-are-zero).
            // Expects well-formed float16 values only (e.g. model weights).
            // Fast Float16 to Float32 Conversion:
            //
            // ┌─[15]─┬─[14]───···───[10]─┬─[9]────····────[0]─┐
            // │ Sign │ Exponent (5 bits) │ Mantissa (10 bits) │ Float16 Layout (16 bits)
            // └──────┴───────────────────┴────────────────────┘
            // │ │ │
            // ▼ ▼ ▼
            // ┌─[31]─┬─[30]───···───[23]─┬─[22]────···────[0]─┐
            // │ Sign │ Exponent (8 bits) │ Mantissa (23 bits) │ Float32 Layout (32 bits)
            // └──────┴───────────────────┴────────────────────┘
            //
            // Shifts and adjustments:
            // - Sign: float16[15] -> float32[31] (shift 16 bits up)
            // - Exponent: float16[10-14] -> float32[23-30] (+ bias adjustment)
            // - Mantissa: float16[0-9] -> float32[13-22] (shift 13 bits up)
            //
            // exp = bits32 & 0x7C00
            // zeroExponentMask = exp == 0 ? 0 : ~0
            var zeroExponentMask = bits32.and(0x7C00).neg().lanewise(VectorOperators.ASHR, 31); // = (-exp) >> 31
            bits32 = bits32.and(0x8000).lanewise(VectorOperators.LSHL, 16) // sign
                    .or(
                            // exponent and mantissa combined
                            bits32.and(0x7FFF).add(0x1C000).lanewise(VectorOperators.LSHL, 13)
                                    .and(zeroExponentMask) // -0, +0 and DAZ (denormals-are-zero)

                    );

            FloatVector thizVector = bits32.reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);
        // Remaining entries.
        if (upperBound < size) {
            result += scalarDot(thiz, thisOffset + upperBound, that, thatOffset + upperBound, size - upperBound);
        }

        return result;
    }
}
