package example.micronaut.utils;

import java.nio.FloatBuffer;
import java.util.stream.IntStream;

import example.micronaut.model.Configuration;
import example.micronaut.model.Llama;
import example.micronaut.model.State;
import example.micronaut.model.Weights;
import example.micronaut.model.tensor.ArrayFloatTensor;
import example.micronaut.model.tensor.FloatTensor;
import lombok.experimental.UtilityClass;

@UtilityClass
public class TransformerUtils {

    public FloatTensor[] allocate(int numTokens, int... dims) {
        return IntStream.range(0, numTokens)
                .mapToObj(i -> ArrayFloatTensor.allocate(dims))
                .toArray(FloatTensor[]::new);
    }

    public void rmsnorm(FloatTensor out, FloatTensor x, FloatBuffer weight, int size, float rmsNormEps) {
        // Calculate sum of squares and normalize in one pass
        float ss = 0f;
        for (int i = 0; i < size; i++) {
            float xi = x.getFloat(i);
            ss += xi * xi;
        }

        float scale = fastInvSqrt(ss / size + rmsNormEps);
        // Normalize and scale
        for (int i = 0; i < size; i++) {
            out.setFloat(i, weight.get(i) * (scale * x.getFloat(i)));
        }
    }

    public FloatTensor forward(Llama model, State state, int[] tokens, int position, boolean computeLogits) {
        Configuration config = model.configuration();
        Weights weights = model.weights();
        int dim = config.dim;
        int headSize = config.headSize;
        int batchSize = state.batchsize;
        int kvDim = (config.dim * config.numberOfKeyValueHeads) / config.numberOfHeads;
        int kvMul = config.numberOfHeads / config.numberOfKeyValueHeads;
        float sqrtHeadSize = (float) fastSqrt(headSize);
        final int nTokens = tokens.length;

        // Copy token embeddings
        Parallel.parallelFor(0, nTokens, batchSize, t -> {
            weights.token_embedding_table.copyTo(tokens[t] * dim, state.x[t], 0, dim);
        });

        // Forward all layers
        for (int l = 0; l < config.numberOfLayers; l++) {
            final int curLayer = l;

            // RMSNorm for attention
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                rmsnorm(state.xb[t], state.x[t], weights.rms_att_weight[curLayer], dim, config.rmsNormEps);
            });

            // QKV matmuls
            weights.wq[l].matmul(nTokens, state.xb, state.q, dim, dim);
            weights.wk[l].matmul(nTokens, state.xb, state.k, kvDim, dim);
            weights.wv[l].matmul(nTokens, state.xb, state.v, kvDim, dim);

            // RoPE relative positional encoding
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                applyRoPE(state.q[t], state.k[t], position + t, headSize, kvDim, weights.freq_cis_real, weights.freq_cis_imag);
            });

            // Save key, value to cache
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                state.k[t].copyTo(0, state.keyCache[curLayer], (position + t) * kvDim, kvDim);
                state.v[t].copyTo(0, state.valueCache[curLayer], (position + t) * kvDim, kvDim);
            });

            // Skip attention and FFN if logits are not required
            if (!computeLogits && curLayer == config.numberOfLayers - 1) {
                state.idxPrevBlock = nTokens - 1;
                return null;
            }

            // Multihead attention
            Parallel.parallelForLong(0, (long) nTokens * config.numberOfHeads, batchSize, ht -> {
                int token = (int) (ht / config.numberOfHeads);
                int h = (int) (ht % config.numberOfHeads);
                int attOffset = h * config.contextLength;
                computeAttention(state, curLayer, token, h, attOffset, position, headSize, kvDim, kvMul, sqrtHeadSize);
            });

            // Final matmul for attention output
            weights.wo[l].matmul(nTokens, state.xb, state.xb2, dim, dim);

            // Residual connection & FFN RMSNorm
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                state.x[t].addInPlace(state.xb2[t]);
                rmsnorm(state.xb[t], state.x[t], weights.rms_ffn_weight[curLayer], dim, config.rmsNormEps);
            });

            // FFN matmuls
            weights.w1[l].matmul(nTokens, state.xb, state.hb, config.hiddenDim, dim);
            weights.w3[l].matmul(nTokens, state.xb, state.hb2, config.hiddenDim, dim);

            // SwiGLU non-linearity
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                applyFFN(state.hb[t], state.hb2[t]);
            });

            // Final FFN matmul
            weights.w2[l].matmul(nTokens, state.hb, state.xb, dim, config.hiddenDim);

            // Residual connection
            Parallel.parallelFor(0, nTokens, batchSize, t -> {
                state.x[t].addInPlace(state.xb[t]);
            });
        }

        // Final RMSNorm
        Parallel.parallelFor(0, nTokens, batchSize, t -> {
            rmsnorm(state.x[t], state.x[t], weights.rms_final_weight, dim, config.rmsNormEps);
        });

        // Classifier into logits
        weights.wcls.matmul(state.x[nTokens - 1], state.logits, config.vocabularySize, dim);
        state.idxPrevBlock = nTokens - 1;

        return state.logits;
    }

    private static void applyRoPE(FloatTensor q, FloatTensor k, int position, int headSize, int kvDim, FloatBuffer freqCisReal, FloatBuffer freqCisImag) {
        for (int i = 0; i < kvDim; i += 2) {
            int headDim = i % headSize;
            float fcr = freqCisReal.get(position * (headSize / 2) + (headDim / 2));
            float fci = freqCisImag.get(position * (headSize / 2) + (headDim / 2));
            applyRotation(q, i, fcr, fci);
            applyRotation(k, i, fcr, fci);
        }
        for (int i = kvDim; i < q.size(); i += 2) {
            int headDim = i % headSize;
            float fcr = freqCisReal.get(position * (headSize / 2) + (headDim / 2));
            float fci = freqCisImag.get(position * (headSize / 2) + (headDim / 2));
            applyRotation(q, i, fcr, fci);
        }
    }

    private static void applyRotation(FloatTensor vec, int i, float fcr, float fci) {
        float v0 = vec.getFloat(i);
        float v1 = vec.getFloat(i + 1);
        vec.setFloat(i, v0 * fcr - v1 * fci);
        vec.setFloat(i + 1, v0 * fci + v1 * fcr);
    }

    private static void computeAttention(State state, int layer, int token, int h, int attOffset, int position, int headSize, int kvDim, int kvMul, float sqrtHeadSize) {
        // Optimize memory access pattern by pre-calculating offsets
        final int qOffset = h * headSize;
        final int baseKeyCacheOffset = (h / kvMul) * headSize;
        final int xbOffset = h * headSize;

        // Pre-calculate scores in a batch
        for (int t = 0; t <= position + token; t++) {
            int keyCacheOffset = t * kvDim + baseKeyCacheOffset;
            float score = state.q[token].dot(qOffset, state.keyCache[layer], keyCacheOffset, headSize) / sqrtHeadSize;
            state.att[token].setFloat(attOffset + t, score);
        }

        state.att[token].softmaxInPlace(attOffset, position + token + 1);
        state.xb[token].fillInPlace(xbOffset, headSize, 0f);
        for (int t = 0; t <= position + token; t++) {
            int vOffset = t * kvDim + (h / kvMul) * headSize;
            float a = state.att[token].getFloat(attOffset + t);
            state.xb[token].saxpyInPlace(xbOffset, state.valueCache[layer], vOffset, headSize, a);
        }
    }

    private static void applyFFN(FloatTensor hb, FloatTensor hb2) {
        final int size = hb.size();
        // Process in chunks for better cache utilization
        final int CHUNK_SIZE = 256;
        for (int chunk = 0; chunk < size; chunk += CHUNK_SIZE) {
            final int end = Math.min(chunk + CHUNK_SIZE, size);
            for (int i = chunk; i < end; i++) {
                float value = hb.getFloat(i);
                value = value / (1f + fastExp(-value));
                hb.setFloat(i, value * hb2.getFloat(i));
            }
        }
    }

    /**
     * Fast inverse square root approximation
     */
    public static float fastSqrt(float x) {
        int i = Float.floatToRawIntBits(x);
        i = ((i + 0x3f800000) >> 1) & 0x7fffffff;
        return Float.intBitsToFloat(i);
    }

    /**
     * Fast inverse square root approximation
     */
    public static float fastInvSqrt(float x) {
        float xhalf = 0.5f * x;
        int i = Float.floatToIntBits(x);
        i = 0x5f3759df - (i >> 1);
        x = Float.intBitsToFloat(i);
        x = x * (1.5f - xhalf * x * x);
        return x;
    }

    /**
     * Fast exponential approximation
     */
    public static float fastExp(float x) {
        x = 1.0f + x / 256.0f;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        x *= x;
        return x;
    }
}
