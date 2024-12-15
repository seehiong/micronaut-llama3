package example.micronaut.utils;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import java.util.Objects;

import example.micronaut.datatype.GGMLTensorEntry;
import example.micronaut.gguf.GGUF;
import example.micronaut.model.Llama;
import example.micronaut.model.PartialModel;
import example.micronaut.model.Weights;
import io.micronaut.context.annotation.Value;
import lombok.experimental.UtilityClass;

/**
 * Support for AOT preloading of GGUF metadata with GraalVM's Native Image.
 *
 * <p>
 * To preload a model at build time, pass
 * {@code -Dllama.PreloadGGUF=/path/to/model.gguf} to the native-image builder
 * command. At runtime, the preloaded model will be used iff the specified and
 * preloaded file names (base name) match.
 */
@UtilityClass
public class AOT {

    @Value("${llama.PreloadGGUF}")
    private String propPreLoadGGUF;

    @Value("${options.max_tokens}")
    private int propMaxTokens;

    private PartialModel preLoaded = null;

    private PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            GGUF gguf = GGUF.loadModel(path);
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                return new PartialModel(
                        path.getFileName().toString(),
                        ModelLoader.loadModel(fileChannel, gguf, propMaxTokens, false),
                        gguf.getTensorDataOffset(),
                        gguf.getTensorInfos());
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Tries to reuse a compatible AOT preloaded model. The file name (base
     * name) must match with the preloaded file name. No checksum/hash is
     * checked for performance reasons.
     */
    public Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        if (preLoaded == null) {
            preLoaded = preLoadGGUF(propPreLoadGGUF);
            if (preLoaded == null) {
                return null; // no pre-loaded model stored
            }
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            // Preloaded and specified model file names didn't match.
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            // Load only the tensors (mmap slices).
            Map<String, GGMLTensorEntry> tensorEntries = TensorUtils.loadTensors(fileChannel,
                    preLoaded.tensorDataOffset(),
                    preLoaded.tensorInfos());
            Weights weights = ModelLoader.loadWeights(tensorEntries, baseModel.configuration());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(),
                    weights);
        }
    }
}
