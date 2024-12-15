package example.micronaut.utils;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.IntConsumer;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import example.micronaut.model.Configuration;
import example.micronaut.model.Llama;
import example.micronaut.model.Pair;
import example.micronaut.model.State;
import example.micronaut.model.Tokenizer;
import example.micronaut.model.Vocabulary;
import example.micronaut.model.tensor.Sampler;
import lombok.experimental.UtilityClass;

@UtilityClass
public class TokenUtils {

    public Tokenizer createTokenizer(Map<String, Object> metadata, Vocabulary vocabulary) {
        String[] mergeLines = (String[]) metadata.get("tokenizer.ggml.merges");
        List<Pair<Integer, Integer>> merges = Arrays.stream(mergeLines)
                .map(line -> line.split(" "))
                .map(parts -> new Pair<>(
                vocabulary.getIndex(parts[0]).orElseThrow(),
                vocabulary.getIndex(parts[1]).orElseThrow()))
                .toList();

        int allTokens = vocabulary.size();
        int baseTokens = 128000; // assume all tokens after the base ones are special.
        List<String> specialTokensList = Arrays.stream(vocabulary.tokens(), baseTokens, allTokens).toList();

        assert specialTokensList.stream().allMatch(token -> vocabulary.getIndex(token).isPresent());

        Map<String, Integer> specialTokens = IntStream.range(0, specialTokensList.size())
                .boxed()
                .collect(Collectors.toMap(
                        i -> specialTokensList.get(i),
                        i -> baseTokens + i));

        return new Tokenizer(vocabulary, merges, ModelLoader.LLAMA_3_PATTERN, specialTokens);
    }

    /**
     * LLM generation entry point, ingest prompt tokens and generates new
     * tokens.
     *
     * <p>
     * All prompt tokens are ingested first, then inference starts, until a stop
     * token is found. The returned tokens only include generated/inferred
     * tokens.
     *
     * @param model model to run inference (including weights, configuration,
     * tokenizer ...)
     * @param state state of the model e.g. key/value caches ... this is mutated
     * by this call
     * @param startPosition start prompt ingestion + inference at this position
     * in the context e.g. useful if state was kept across calls (chained
     * generation). 0 implies run with no previous context.
     * @param promptTokens prompt tokens to ingest, all the prompt tokens will
     * be ingested, given there's enough capacity left in the context
     * @param stopTokens set of tokens that abort generation during inference,
     * stop tokens do not affect prompt ingestion
     * @param maxTokens maximum number of tokens (can go up to
     * {@link Configuration#contextLength context length} if this value is
     * negative or greater than
     * {@link Configuration#contextLength context length}
     * @param sampler {@link Sampler strategy} used to select tokens
     * @param echo debugging flag, prints ALL, prompt and inferred tokens, to
     * {@link System#err stderr}
     * @param onTokenGenerated callback, if non-null, it's called every time a
     * token is inferred e.g. it's not called when ingesting prompt tokens
     * @return list of generated/inferred tokens, including the stop token, if
     * any e.g. does not include any token from the prompt
     */
    public List<Integer> generateTokens(Llama model, State state, int startPosition, List<Integer> promptTokens,
            Set<Integer> stopTokens, int maxTokens, Sampler sampler, boolean echo,
            IntConsumer onTokenGenerated) {
        long startNanos = System.nanoTime();
        long startGen = 0;
        if (maxTokens < 0 || model.configuration().contextLength < maxTokens) {
            maxTokens = model.configuration().contextLength;
        }
        List<Integer> generatedTokens = new ArrayList<>(maxTokens);
        int token = state.latestToken; // BOS?
        int nextToken;
        int promptIndex = 0;
        for (int position = startPosition; position < maxTokens; ++position) {
            if (promptIndex < promptTokens.size()) {
                final int nTokens = Math.min(maxTokens - position,
                        Math.min(promptTokens.size() - promptIndex, state.batchsize));
                final int[] tokens = new int[nTokens];
                for (int i = 0; i < nTokens; i++) {
                    tokens[i] = promptTokens.get(promptIndex + i);
                    if (echo) {
                        // log prompt token (different color?)
                        System.err.print(
                                Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(tokens[i]))));
                    }
                }
                if (echo) {
                    System.out.format("position=%d, promptIdx=%d, promptSize=%d, tokens=%s%n", position, promptIndex,
                            promptTokens.size(), Arrays.toString(tokens));
                }
                // Only compute logits on the very last batch.
                boolean computeLogits = promptIndex + nTokens >= promptTokens.size();
                TransformerUtils.forward(model, state, tokens, position, computeLogits);
                position += nTokens - 1; // -1 -> incremented later in the for loop
                promptIndex += nTokens;
                if (promptIndex < promptTokens.size()) {
                    continue;
                }
                startGen = System.nanoTime();
            } else {
                TransformerUtils.forward(model, state, new int[]{token}, position, true);
            }
            nextToken = sampler.sampleToken(state.logits);
            if (echo) {
                // log inferred token
                System.err.print(Tokenizer.replaceControlCharacters(model.tokenizer().decode(List.of(nextToken))));
            }
            generatedTokens.add(nextToken);
            if (onTokenGenerated != null) {
                onTokenGenerated.accept(nextToken);
            }
            if (stopTokens.contains(nextToken)) {
                break;
            }
            state.latestToken = token = nextToken;
        }

        long elapsedNanos = System.nanoTime() - startNanos;
        long promptNanos = startGen - startNanos;
        long genNanos = elapsedNanos - startGen + startNanos;
        System.err.printf("%nprompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%n",
                promptTokens.size() / (promptNanos / 1_000_000_000.0), promptTokens.size(),
                generatedTokens.size() / (genNanos / 1_000_000_000.0), generatedTokens.size());

        return generatedTokens;
    }
}
