package example.micronaut.utils;

import java.util.function.IntConsumer;
import java.util.function.LongConsumer;
import java.util.stream.IntStream;
import java.util.stream.LongStream;

import lombok.experimental.UtilityClass;

@UtilityClass
public class Parallel {

    public void parallelFor(int startInclusive, int endExclusive, IntConsumer action) {
        IntStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public void parallelForLong(long startInclusive, long endExclusive, LongConsumer action) {
        LongStream.range(startInclusive, endExclusive).parallel().forEach(action);
    }

    public void parallelFor(int startInclusive, int endExclusive, int batchSize, IntConsumer action) {
        int totalElements = endExclusive - startInclusive;
        int numBatches = (totalElements + batchSize - 1) / batchSize;

        IntStream.range(0, numBatches).parallel().forEach(batch -> {
            int start = startInclusive + batch * batchSize;
            int end = Math.min(start + batchSize, endExclusive);
            for (int i = start; i < end; i++) {
                action.accept(i);
            }
        });
    }

    public void parallelForLong(long startInclusive, long endExclusive, long batchSize, LongConsumer action) {
        long totalElements = endExclusive - startInclusive;
        long numBatches = (totalElements + batchSize - 1) / batchSize;

        LongStream.range(0, numBatches).parallel().forEach(batch -> {
            long start = startInclusive + batch * batchSize;
            long end = Math.min(start + batchSize, endExclusive);
            for (long i = start; i < end; i++) {
                action.accept(i);
            }
        });
    }
}
