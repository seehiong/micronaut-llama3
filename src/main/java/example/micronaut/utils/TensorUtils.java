package example.micronaut.utils;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.function.IntFunction;

import example.micronaut.gguf.GGMLTensorEntry;
import example.micronaut.gguf.GGMLType;
import example.micronaut.gguf.GGUFTensorInfo;
import example.micronaut.model.tensor.BF16FloatTensor;
import example.micronaut.model.tensor.F16FloatTensor;
import example.micronaut.model.tensor.FloatTensor;
import example.micronaut.model.tensor.Q4_0FloatTensor;
import example.micronaut.model.tensor.Q4_KFloatTensor;
import example.micronaut.model.tensor.Q6_KFloatTensor;
import example.micronaut.model.tensor.Q8_0FloatTensor;
import lombok.experimental.UtilityClass;

@UtilityClass
public class TensorUtils {

    public Map<String, GGMLTensorEntry> loadTensors(FileChannel fileChannel, long tensorDataOffset,
            Map<String, GGUFTensorInfo> tensorInfos) throws IOException {
        Arena arena = Arena.ofAuto();
        MemorySegment tensorData = fileChannel.map(FileChannel.MapMode.READ_ONLY, tensorDataOffset,
                fileChannel.size() - tensorDataOffset, arena);
        Map<String, GGMLTensorEntry> tensorEntries = HashMap.newHashMap(tensorInfos.size());
        for (Map.Entry<String, GGUFTensorInfo> entry : tensorInfos.entrySet()) {
            GGUFTensorInfo ti = entry.getValue();
            int numberOfElements = numberOfElements(ti.dimensions());
            int sizeInBytes = Math.toIntExact(ti.ggmlType().byteSizeFor(numberOfElements));
            MemorySegment memorySegment = tensorData.asSlice(ti.offset(), sizeInBytes);
            tensorEntries.put(ti.name(),
                    new GGMLTensorEntry(tensorData, ti.name(), ti.ggmlType(), ti.dimensions(), memorySegment));
        }
        return tensorEntries;
    }

    public FloatTensor loadQuantized(GGMLTensorEntry entry) {
        GGMLType ggmlType = entry.ggmlType();
        return switch (ggmlType) {
            case Q8_0 ->
                new Q8_0FloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_0 ->
                new Q4_0FloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            case Q4_K ->
                new Q4_KFloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            case Q6_K ->
                new Q6_KFloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            case BF16 ->
                new BF16FloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            case F16 ->
                new F16FloatTensor(numberOfElements(entry.shape()), entry.memorySegment());
            // case F32 -> new F32FloatTensor(numberOfElements(entry.shape()),
            // entry.memorySegment());
            default ->
                throw new UnsupportedOperationException("Quantization format " + ggmlType);
        };
    }

    public FloatTensor[] loadArrayOfQuantized(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatTensor[] array = new FloatTensor[size];
        for (int i = 0; i < size; i++) {
            array[i] = loadQuantized(getTensorEntry.apply(i));
        }
        return array;
    }

    public FloatBuffer[] loadArrayOfFloatBuffer(int size, IntFunction<GGMLTensorEntry> getTensorEntry) {
        FloatBuffer[] array = new FloatBuffer[size];
        for (int i = 0; i < size; i++) {
            array[i] = toFloatBuffer(getTensorEntry.apply(i));
        }
        return array;
    }

    public FloatBuffer toFloatBuffer(GGMLTensorEntry tensorEntry) {
        GGMLType ggmlType = tensorEntry.ggmlType();
        return switch (ggmlType) {
            case F32 ->
                tensorEntry.memorySegment().asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            default ->
                throw new UnsupportedOperationException("Conversion to " + ggmlType);
        };
    }

    public int numberOfElements(int... dimensions) {
        assert Arrays.stream(dimensions).allMatch(i -> i > 0);
        return Arrays.stream(dimensions).reduce(Math::multiplyExact).orElseThrow();
    }

}
