package example.micronaut.model;

import java.util.stream.Stream;

import example.micronaut.model.tensor.ArrayFloatTensor;
import example.micronaut.model.tensor.FloatTensor;
import example.micronaut.utils.TransformerUtils;

public class State {

    // current wave of activations
    public final int batchsize;
    public final FloatTensor[] x; // activation at current time stamp (dim,)
    public final FloatTensor[] xb; // same, but inside a residual branch (dim,)
    public final FloatTensor[] xb2; // an additional buffer just for convenience (dim,)
    public final FloatTensor[] hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor[] hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    public final FloatTensor[] q; // query (dim,)
    public final FloatTensor[] k; // key (dim,)
    public final FloatTensor[] v; // value (dim,)
    public final FloatTensor[] att; // buffer for scores/attention values (n_heads, seq_len)
    public final FloatTensor logits; // output logits

    // kv cache
    public final FloatTensor[] keyCache; // (n_layer, seq_len, kv_dim)
    public final FloatTensor[] valueCache; // (n_layer, seq_len, kv_dim)

    /**
     * last index in previous block
     */
    public int idxPrevBlock;

    public int latestToken;

    State(Configuration config, int batchsize) {
        this.batchsize = batchsize;
        this.x = TransformerUtils.allocate(batchsize, config.dim);
        this.xb = TransformerUtils.allocate(batchsize, config.dim);
        this.xb2 = TransformerUtils.allocate(batchsize, config.dim);
        this.hb = TransformerUtils.allocate(batchsize, config.hiddenDim);
        this.hb2 = TransformerUtils.allocate(batchsize, config.hiddenDim);
        this.q = TransformerUtils.allocate(batchsize, config.dim);
        this.k = TransformerUtils.allocate(batchsize, config.dim);
        this.v = TransformerUtils.allocate(batchsize, config.dim);
        this.att = TransformerUtils.allocate(batchsize, config.numberOfHeads, config.contextLength);
        idxPrevBlock = -1;

        this.logits = ArrayFloatTensor.allocate(config.vocabularySize);
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        this.keyCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim))
                .limit(config.numberOfLayers).toArray(FloatTensor[]::new);
        this.valueCache = Stream.generate(() -> ArrayFloatTensor.allocate(config.contextLength, kvDim))
                .limit(config.numberOfLayers).toArray(FloatTensor[]::new);
    }
}
