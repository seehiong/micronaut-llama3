package example.micronaut.gguf;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import example.micronaut.model.Pair;
import example.micronaut.utils.Timer;

public class GGUF {

    private static final int GGUF_MAGIC = 0x46554747;
    private static final int DEFAULT_ALIGNMENT = 32; // must be a power of 2
    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int magic;
    private int version;
    private int tensorCount; // uint64_t
    private int alignment;
    private int metadata_kv_count; // uint64_t
    private Map<String, Object> metadata;

    public Map<String, GGUFTensorInfo> getTensorInfos() {
        return tensorInfos;
    }

    private Map<String, GGUFTensorInfo> tensorInfos;

    private long tensorDataOffset;

    public long getTensorDataOffset() {
        return tensorDataOffset;
    }

    public Map<String, Object> getMetadata() {
        return metadata;
    }

    private final ByteBuffer buffer = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);

    public static GGUF loadModel(Path modelPath) throws IOException {
        try (FileChannel fileChannel = FileChannel.open(modelPath); var ignored = Timer.log("Parse " + modelPath)) {
            GGUF gguf = new GGUF();
            gguf.loadModelImpl(fileChannel);
            return gguf;
        }
    }

    private void loadModelImpl(FileChannel fileChannel) throws IOException {
        // The header of the file.
        readHeader(fileChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        this.tensorInfos = HashMap.newHashMap(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            GGUFTensorInfo ti = readTensorInfo(fileChannel);
            assert !tensorInfos.containsKey(ti.name());
            tensorInfos.put(ti.name(), ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        // long _padding = -fileChannel.position() & (ALIGNMENT - 1);
        long _padding = getAlignment() - (fileChannel.position() % getAlignment());
        fileChannel.position(fileChannel.position() + _padding);
        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This
        // data should be close
        // or identical to the data in the original model file, but may be different due
        // to quantization or
        // other optimizations for inference. Any such deviations should be recorded in
        // the metadata or as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its
        // `tensor_infos` entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the
        // space between tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        this.tensorDataOffset = fileChannel.position();
    }

    private GGMLType readGGMLType(FileChannel fileChannel) throws IOException {
        int ggmlTypeId = readInt(fileChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private GGUFTensorInfo readTensorInfo(FileChannel fileChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(fileChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of dimensions in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(fileChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The dimensions of the tensor.
        int[] dimensions = new int[n_dimensions]; // uint64_t dimensions[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = Math.toIntExact(readLong(fileChannel));
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(fileChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(fileChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    private String readString(FileChannel fileChannel) throws IOException {
        int len = Math.toIntExact(readLong(fileChannel));
        ByteBuffer stringBuffer = ByteBuffer.allocate(len);
        int bytesRead = fileChannel.read(stringBuffer);
        assert len == bytesRead;
        return new String(stringBuffer.array(), StandardCharsets.UTF_8);
    }

    private Pair<String, Object> readKeyValuePair(FileChannel fileChannel) throws IOException {
        // The key of the metadata. It is a standard GGUF string, with the following
        // caveats:
        // - It must be a valid ASCII string.
        // - It must be a hierarchical key, where each segment is `lower_snake_case` and
        // separated by a `.`.
        // - It must be at most 2^16-1/65535 bytes long.
        // Any keys that do not follow these rules are invalid.
        String key = readString(fileChannel); // gguf_string_t key;
        assert key.length() < (1 << 16);
        assert key.codePoints()
                .allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
        Object value = readMetadataValue(fileChannel);
        return new Pair<>(key, value);
    }

    private Object readMetadataValue(FileChannel fileChannel) throws IOException {
        // The type of the value.
        // Must be one of the `gguf_metadata_value_type` values.
        MetadataValueType value_type = readMetadataValueType(fileChannel); // gguf_metadata_value_type value_type;
        // The value.
        return readMetadataValueOfType(value_type, fileChannel); // gguf_metadata_value_t value;
    }

    private void readHeader(FileChannel fileChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        this.magic = readInt(fileChannel); // uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("unsupported header.magic " + magic);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update
        // the metadata
        // to signify the change.
        this.version = readInt(fileChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException("unsupported header.version " + version);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is
        // always present
        // for loading the tensors.
        this.tensorCount = Math.toIntExact(readLong(fileChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        this.metadata_kv_count = Math.toIntExact(readLong(fileChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = HashMap.newHashMap(metadata_kv_count);
        for (int i = 0; i < metadata_kv_count; ++i) {
            Pair<String, Object> keyValue = readKeyValuePair(fileChannel);
            assert !metadata.containsKey(keyValue.first());
            metadata.put(keyValue.first(), keyValue.second());
        }
    }

    private Object readArray(FileChannel fileChannel) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType valueType = readMetadataValueType(fileChannel); // gguf_metadata_value_type type;
        // Number of elements, not bytes
        int len = Math.toIntExact(readLong(fileChannel)); // uint64_t len;

        return switch (valueType) {
            case UINT8, INT8 ->
                readByteArray(fileChannel, len);
            case UINT16, INT16 ->
                readShortArray(fileChannel, len);
            case UINT32, INT32 ->
                readIntArray(fileChannel, len);
            case FLOAT32 ->
                readFloatArray(fileChannel, len);
            case BOOL ->
                readBooleanArray(fileChannel, len);
            case STRING ->
                readStringArray(fileChannel, len);
            case ARRAY ->
                readNestedArray(fileChannel, len);
            default ->
                throw new UnsupportedOperationException("Read array of " + valueType);
        };
    }

    private byte[] readByteArray(FileChannel fileChannel, int len) throws IOException {
        byte[] bytes = new byte[len];
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);
        fileChannel.read(byteBuffer);
        return bytes;
    }

    private short[] readShortArray(FileChannel fileChannel, int len) throws IOException {
        short[] shorts = new short[len];
        ByteBuffer shortBuffer = ByteBuffer.allocate(len * Short.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        fileChannel.read(shortBuffer);
        shortBuffer.flip();
        shortBuffer.asShortBuffer().get(shorts);
        return shorts;
    }

    private int[] readIntArray(FileChannel fileChannel, int len) throws IOException {
        int[] ints = new int[len];
        ByteBuffer intBuffer = ByteBuffer.allocate(len * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        fileChannel.read(intBuffer);
        intBuffer.flip();
        intBuffer.asIntBuffer().get(ints);
        return ints;
    }

    private float[] readFloatArray(FileChannel fileChannel, int len) throws IOException {
        float[] floats = new float[len];
        ByteBuffer floatBuffer = ByteBuffer.allocate(len * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        fileChannel.read(floatBuffer);
        floatBuffer.flip();
        floatBuffer.asFloatBuffer().get(floats);
        return floats;
    }

    private boolean[] readBooleanArray(FileChannel fileChannel, int len) throws IOException {
        boolean[] booleans = new boolean[len];
        byte[] bytes = new byte[len];
        ByteBuffer byteBuffer = ByteBuffer.wrap(bytes);
        fileChannel.read(byteBuffer);
        for (int i = 0; i < len; i++) {
            booleans[i] = bytes[i] != 0;
        }
        return booleans;
    }

    private String[] readStringArray(FileChannel fileChannel, int len) throws IOException {
        String[] strings = new String[len];
        for (int i = 0; i < len; i++) {
            strings[i] = readString(fileChannel);
        }
        return strings;
    }

    private Object[] readNestedArray(FileChannel fileChannel, int len) throws IOException {
        Object[] arrays = new Object[len];
        for (int i = 0; i < len; i++) {
            arrays[i] = readArray(fileChannel);
        }
        return arrays;
    }

    private Object readMetadataValueOfType(MetadataValueType valueType, FileChannel fileChannel) throws IOException {
        return switch (valueType) {
            case UINT8, INT8 ->
                readByte(fileChannel);
            case UINT16, INT16 ->
                readShort(fileChannel);
            case UINT32, INT32 ->
                readInt(fileChannel);
            case FLOAT32 ->
                readFloat(fileChannel);
            case UINT64, INT64 ->
                readLong(fileChannel);
            case FLOAT64 ->
                readDouble(fileChannel);
            case BOOL ->
                readBoolean(fileChannel);
            case STRING ->
                readString(fileChannel);
            case ARRAY ->
                readArray(fileChannel);
        };
    }

    private byte readByte(FileChannel fileChannel) throws IOException {
        buffer.clear().limit(Byte.BYTES);
        int bytesRead = fileChannel.read(buffer);
        assert bytesRead == Byte.BYTES;
        return buffer.flip().get();
    }

    private boolean readBoolean(FileChannel fileChannel) throws IOException {
        return readByte(fileChannel) != 0;
    }

    private short readShort(FileChannel fileChannel) throws IOException {
        buffer.clear().limit(Short.BYTES);
        int bytesRead = fileChannel.read(buffer);
        assert bytesRead == Short.BYTES;
        return buffer.flip().getShort();
    }

    private int readInt(FileChannel fileChannel) throws IOException {
        buffer.clear().limit(Integer.BYTES);
        int bytesRead = fileChannel.read(buffer);
        assert bytesRead == Integer.BYTES;
        return buffer.flip().getInt();
    }

    private long readLong(FileChannel fileChannel) throws IOException {
        buffer.clear().limit(Long.BYTES);
        int bytesRead = fileChannel.read(buffer);
        assert bytesRead == Long.BYTES;
        return buffer.flip().getLong();
    }

    private float readFloat(FileChannel fileChannel) throws IOException {
        return Float.intBitsToFloat(readInt(fileChannel));
    }

    private double readDouble(FileChannel fileChannel) throws IOException {
        return Double.longBitsToDouble(readLong(fileChannel));
    }

    private MetadataValueType readMetadataValueType(FileChannel fileChannel) throws IOException {
        int index = readInt(fileChannel);
        return MetadataValueType.fromIndex(index);
    }

    public int getAlignment() {
        if (alignment == 0) {
            alignment = (int) metadata.getOrDefault("general.alignment", DEFAULT_ALIGNMENT);
            assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        }
        return alignment;
    }
}
