package example.micronaut.model;

public record Llama(Configuration configuration, Tokenizer tokenizer, Weights weights) {
    public State createNewState(int batchsize) {
        State state = new State(configuration(), batchsize);
        state.latestToken = tokenizer.getSpecialTokens().get("<|begin_of_text|>");
        return state;
    }
}
