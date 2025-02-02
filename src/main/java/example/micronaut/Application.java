package example.micronaut;

import io.micronaut.context.ApplicationContext;
import io.micronaut.context.annotation.Value;
import io.micronaut.runtime.Micronaut;
import jakarta.inject.Singleton;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@Singleton
public class Application {

    private final String parallelism;
    private final String batchSize;
    private final String vectorBitSize;

    public Application(@Value("${java.util.concurrent.ForkJoinPool.common.parallelism:16}") String parallelism,
            @Value("${llama.BatchSize}") String batchSize,
            @Value("${llama.VectorBitSize}") String vectorBitSize) {
        this.parallelism = parallelism;
        this.batchSize = batchSize;
        this.vectorBitSize = vectorBitSize;
    }

    public void run(String[] args) {
        System.getProperties().putIfAbsent("java.util.concurrent.ForkJoinPool.common.parallelism", parallelism);
        System.getProperties().putIfAbsent("llama.BatchSize", batchSize);
        System.getProperties().putIfAbsent("llama.VectorBitSize", vectorBitSize);

        log.info("ForkJoinPool parallelism: "
                + System.getProperty("java.util.concurrent.ForkJoinPool.common.parallelism"));
        log.info("llama.BatchSize: " + System.getProperty("llama.BatchSize"));
        log.info("llama.VectorBitSize: " + System.getProperty("llama.VectorBitSize"));
    }

    public static void main(String[] args) {
        ApplicationContext context = Micronaut.run(Application.class, args);
        Application app = context.getBean(Application.class);
        app.run(args);
    }
}
