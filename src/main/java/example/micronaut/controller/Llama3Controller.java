package example.micronaut.controller;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import example.micronaut.model.Llama;
import example.micronaut.model.LlamaOptions;
import example.micronaut.model.tensor.Sampler;
import example.micronaut.service.Llama3Service;
import example.micronaut.utils.AOT;
import example.micronaut.utils.ModelLoader;
import example.micronaut.utils.SamplingUtils;
import io.micronaut.context.annotation.Value;
import io.micronaut.http.MediaType;
import io.micronaut.http.annotation.Controller;
import io.micronaut.http.annotation.Get;
import io.micronaut.http.annotation.QueryValue;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import reactor.core.publisher.Flux;

@Controller("/api/llama3")
@RequiredArgsConstructor
public class Llama3Controller {

    private final Llama3Service llama3Service;

    @Value("${options.model_path}")
    private String propModelPath;

    @Value("${options.temperature}")
    private float propTemperature;

    @Value("${options.topp}")
    private float propTopp;

    @Value("${options.seed}")
    private long propSeed;

    @Value("${options.max_tokens}")
    private int propMaxTokens;

    @Value("${options.stream}")
    private boolean propStream;

    @Value("${options.echo}")
    private boolean propEcho;

    private Llama model;
    private Sampler sampler;
    private LlamaOptions options;

    @PostConstruct
    public void init() throws IOException {
        if (propSeed == -1) {
            propSeed = System.nanoTime();
        }
        Path modelPath = Paths.get(propModelPath);

        model = AOT.tryUsePreLoaded(modelPath, propMaxTokens);
        if (model == null) {
            // No compatible preloaded model found, fallback to fully parse and load the
            // specified file.
            model = ModelLoader.loadModel(modelPath, propMaxTokens, true);
        }
        sampler = SamplingUtils.selectSampler(model.configuration().vocabularySize, propTemperature, propTopp, propSeed);

        options = new LlamaOptions(modelPath, null, null, true, propTemperature, propTopp, propSeed, propMaxTokens,
                propStream, propEcho);
    }

    @Get(value = "/generate", produces = MediaType.TEXT_EVENT_STREAM)
    public Flux<Object> generate(@QueryValue(defaultValue = "Once upon a time") String prompt) {
        options.setPrompt(prompt);
        options.setInteractive(false);
        return llama3Service.runInstructOnce(model, sampler, options);
    }

    @Get(value = "/chat", produces = MediaType.TEXT_EVENT_STREAM)
    public Flux<Object> chat(@QueryValue(defaultValue = "Once upon a time") String prompt,
            @QueryValue(defaultValue = "You are a helpful assistant.") String system_prompt) {
        options.setPrompt(prompt);
        options.setSystemPrompt(system_prompt);
        options.setInteractive(true);
        return llama3Service.runInteractive(model, sampler, options);
    }

}
