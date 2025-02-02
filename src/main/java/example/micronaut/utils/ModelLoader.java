package example.micronaut.utils;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

import example.micronaut.gguf.GGMLTensorEntry;
import example.micronaut.gguf.GGUF;
import example.micronaut.model.Configuration;
import example.micronaut.model.Llama;
import example.micronaut.model.Pair;
import example.micronaut.model.Tokenizer;
import example.micronaut.model.Vocabulary;
import example.micronaut.model.Weights;
import lombok.experimental.UtilityClass;

@UtilityClass
public class ModelLoader {

    private static final String TOKENIZER_LLAMA_3_MODEL = "gpt2";

    public static final String LLAMA_3_PATTERN = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    private Vocabulary loadVocabulary(Map<String, Object> metadata) {
        String model = (String) metadata.get("tokenizer.ggml.model");
        if (!TOKENIZER_LLAMA_3_MODEL.equals(model)) {
            throw new IllegalArgumentException(
                    "expected " + TOKENIZER_LLAMA_3_MODEL + " but found " + model);
        }
        String[] tokens = (String[]) metadata.get("tokenizer.ggml.tokens");
        return new Vocabulary(tokens, null);
    }

    public Llama loadModel(Path ggufPath, int contextLength, boolean loadWeights) throws IOException {
        GGUF gguf = GGUF.loadModel(ggufPath);
        FileChannel fileChannel = FileChannel.open(ggufPath, StandardOpenOption.READ);
        return loadModel(fileChannel, gguf, contextLength, loadWeights);
    }

    public Llama loadModel(FileChannel fileChannel, GGUF gguf, int contextLength, boolean loadWeights)
            throws IOException {
        try (var ignored = Timer.log("Load LlaMa model")) {
            Map<String, Object> metadata = gguf.getMetadata();
            Vocabulary vocabulary = loadVocabulary(metadata);
            Tokenizer tokenizer = TokenUtils.createTokenizer(metadata, vocabulary);

            Configuration config = new Configuration(
                    (int) metadata.get("llama.embedding_length"),
                    (int) metadata.get("llama.feed_forward_length"),
                    (int) metadata.get("llama.block_count"),
                    (int) metadata.get("llama.attention.head_count"),
                    metadata.containsKey("llama.attention.head_count_kv")
                    ? (int) metadata.get("llama.attention.head_count_kv")
                    : (int) metadata.get("llama.attention.head_count"),
                    vocabulary.size(),
                    (int) metadata.get("llama.context_length"),
                    (float) metadata.getOrDefault("llama.attention.layer_norm_rms_epsilon", 1e-5f),
                    (float) metadata.getOrDefault("llama.rope.freq_base", 10000f))
                    .withContextLength(contextLength);

            Weights weights = null;
            if (loadWeights) {
                Map<String, GGMLTensorEntry> tensorEntries = TensorUtils.loadTensors(fileChannel,
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos());
                weights = loadWeights(tensorEntries, config);
            }
            return new Llama(config, tokenizer, weights);
        }
    }

    public Weights loadWeights(Map<String, GGMLTensorEntry> tensorEntries, Configuration config) {
        boolean ropeScaling = tensorEntries.containsKey("rope_freqs");
        float scaleFactor = 8;
        float loFreqFactor = 1;
        float hiFreqFactor = 3;
        int oldContextLength = 8192;
        Pair<float[], float[]> ropeFreqs = RoPE.precomputeFreqsCis(config.contextLength, config.headSize,
                config.ropeTheta,
                ropeScaling, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
        float[] ropeFreqsReal = ropeFreqs.first();
        float[] ropeFreqsImag = ropeFreqs.second();

        GGMLTensorEntry tokenEmbeddings = tensorEntries.get("token_embd.weight");
        Weights qw = new Weights(
                TensorUtils.loadQuantized(tokenEmbeddings),
                TensorUtils.loadArrayOfFloatBuffer(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".attn_norm.weight")),
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".attn_q.weight")),
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".attn_k.weight")),
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".attn_v.weight")),
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".attn_output.weight")),
                TensorUtils.loadArrayOfFloatBuffer(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".ffn_norm.weight")),
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".ffn_gate.weight")), // w1
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".ffn_down.weight")), // w2
                TensorUtils.loadArrayOfQuantized(config.numberOfLayers,
                        i -> tensorEntries.get("blk." + i + ".ffn_up.weight")), // w3
                TensorUtils.toFloatBuffer(tensorEntries.get("output_norm.weight")),
                FloatBuffer.wrap(ropeFreqsReal),
                FloatBuffer.wrap(ropeFreqsImag),
                // If "output.weight" is not present then the embedding weights are tied/shared
                // with the decoder.
                // This is commonly referred as "tie word embeddings".
                TensorUtils.loadQuantized(
                        tensorEntries.getOrDefault("output.weight", tokenEmbeddings)));
        return qw;
    }
}
