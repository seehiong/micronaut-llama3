package example.micronaut.model;

import java.nio.file.Path;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class LlamaOptions {

    Path modelPath;
    String prompt;
    String systemPrompt;
    boolean interactive;
    float temperature;
    float topp;
    long seed;
    int maxTokens;
    boolean stream;
    boolean echo;
}
