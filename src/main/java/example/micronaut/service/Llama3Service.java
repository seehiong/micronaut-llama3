package example.micronaut.service;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

import example.micronaut.model.ChatFormat;
import example.micronaut.model.Llama;
import example.micronaut.model.LlamaOptions;
import example.micronaut.model.State;
import example.micronaut.model.tensor.Sampler;
import example.micronaut.utils.TokenUtils;
import io.micronaut.context.annotation.Value;
import jakarta.inject.Singleton;
import reactor.core.publisher.Flux;
import reactor.core.scheduler.Schedulers;

@Singleton
public class Llama3Service {

    @Value("${llama.BatchSize}")
    private int propBatchSize;

    public Flux<Object> runInteractive(Llama model, Sampler sampler, LlamaOptions options) {
        return Flux.create(emitter -> {

            State state = null;
            List<Integer> conversationTokens = new ArrayList<>();
            ChatFormat chatFormat = new ChatFormat(model.tokenizer());
            conversationTokens.add(chatFormat.beginOfText);
            if (options.getSystemPrompt() != null) {
                conversationTokens.addAll(
                        chatFormat
                                .encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.getSystemPrompt())));
            }
            int startPosition = 0;

            if (state == null) {
                state = model.createNewState(propBatchSize);
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.getPrompt())));
            conversationTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));
            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = TokenUtils.generateTokens(model, state, startPosition,
                    conversationTokens.subList(startPosition, conversationTokens.size()), stopTokens,
                    options.getMaxTokens(), sampler, options.isEcho(), token -> {
                if (options.isStream()) {
                    if (!model.tokenizer().isSpecialToken(token)) {
                        emitter.next(model.tokenizer().decode(List.of(token)));
                    }
                }
            });
            // Include stop token in the prompt history, but not in the response displayed
            // to the user.
            conversationTokens.addAll(responseTokens);
            startPosition = conversationTokens.size();
            Integer stopToken = null;
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                stopToken = responseTokens.getLast();
                responseTokens.removeLast();
            }
            if (stopToken == null) {
                emitter.next("Ran out of context length...");
            }

            emitter.complete();
        }).subscribeOn(Schedulers.boundedElastic());
    }

    public Flux<Object> runInstructOnce(Llama model, Sampler sampler, LlamaOptions options) {
        return Flux.create(emitter -> {

            State state = model.createNewState(propBatchSize);
            ChatFormat chatFormat = new ChatFormat(model.tokenizer());
            List<Integer> promptTokens = new ArrayList<>();
            promptTokens.add(chatFormat.beginOfText);
            if (options.getSystemPrompt() != null) {
                promptTokens.addAll(
                        chatFormat
                                .encodeMessage(new ChatFormat.Message(ChatFormat.Role.SYSTEM, options.getSystemPrompt())));
            }
            promptTokens
                    .addAll(chatFormat.encodeMessage(new ChatFormat.Message(ChatFormat.Role.USER, options.getPrompt())));
            promptTokens.addAll(chatFormat.encodeHeader(new ChatFormat.Message(ChatFormat.Role.ASSISTANT, "")));

            Set<Integer> stopTokens = chatFormat.getStopTokens();
            List<Integer> responseTokens = TokenUtils.generateTokens(model, state, 0, promptTokens, stopTokens,
                    options.getMaxTokens(), sampler, options.isEcho(), token -> {
                if (options.isStream()) {
                    if (!model.tokenizer().isSpecialToken(token)) {
                        emitter.next(model.tokenizer().decode(List.of(token)));
                    }
                }
            });
            if (!responseTokens.isEmpty() && stopTokens.contains(responseTokens.getLast())) {
                responseTokens.removeLast();
            }
            if (!options.isStream()) {
                String responseText = model.tokenizer().decode(responseTokens);
                emitter.next(responseText);
            }

            emitter.complete();
        }).subscribeOn(Schedulers.boundedElastic());
    }
}
