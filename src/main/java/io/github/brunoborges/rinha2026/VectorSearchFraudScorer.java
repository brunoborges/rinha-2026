package io.github.brunoborges.rinha2026;

import java.util.Arrays;
import java.util.concurrent.Semaphore;

/**
 * {@link FraudScorer} backed by brute-force K-nearest-neighbor search over the
 * labeled {@link ReferenceDataset}, as specified in
 * {@code docs/DETECTION_RULES.md}.
 *
 * <p>For each request the transaction is vectorized, the {@value #K} nearest
 * reference vectors (by Euclidean distance) are found, and the fraud score is
 * the fraction of those neighbors labeled fraud. A transaction is approved when
 * its score is below {@value #THRESHOLD}.
 *
 * <p>The scan is O(count &times; dimensions) per request. Because the challenge
 * caps the solution at one CPU unit, concurrent scans only thrash the cache, so
 * scoring is bounded by a {@link Semaphore} sized to the available processors;
 * HTTP concurrency (virtual threads) is unaffected.
 */
public final class VectorSearchFraudScorer implements FraudScorer {

    /** Number of nearest neighbors considered. */
    public static final int K = 5;

    /** Approval threshold: approved when {@code fraud_score < THRESHOLD}. */
    public static final double THRESHOLD = 0.6;

    private final TransactionVectorizer vectorizer;
    private final ReferenceDataset dataset;
    private final Semaphore scanPermits;

    /**
     * @param vectorizer the transaction vectorizer
     * @param dataset    the reference dataset to search (must be non-empty)
     */
    public VectorSearchFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset) {
        if (dataset.count() == 0) {
            throw new IllegalArgumentException("reference dataset is empty");
        }
        this.vectorizer = vectorizer;
        this.dataset = dataset;
        this.scanPermits = new Semaphore(Math.max(1, Runtime.getRuntime().availableProcessors()));
    }

    @Override
    public FraudResponse score(FraudRequest request) {
        short[] query = ReferenceDataset.quantize(vectorizer.vectorize(request));
        int frauds = countFraudsAmongNearest(query);
        int k = Math.min(K, dataset.count());
        double fraudScore = (double) frauds / k;
        boolean approved = fraudScore < THRESHOLD;
        return new FraudResponse(approved, fraudScore);
    }

    @Override
    public void preload() {
        dataset.preload();
    }

    /**
     * Finds the {@code min(K, count)} nearest references to {@code query} and
     * returns how many are labeled fraud. Maintains a fixed-size buffer of the
     * smallest squared distances seen so far; ties keep the earlier record
     * (strict {@code <} replacement).
     */
    private int countFraudsAmongNearest(short[] query) {
        int count = dataset.count();
        int k = Math.min(K, count);

        long[] bestDist = new long[k];
        boolean[] bestFraud = new boolean[k];
        Arrays.fill(bestDist, Long.MAX_VALUE);
        int worst = 0; // index of the current largest distance in bestDist

        scanPermits.acquireUninterruptibly();
        try {
            for (int i = 0; i < count; i++) {
                long d2 = dataset.squaredDistance(query, i);
                if (d2 < bestDist[worst]) {
                    bestDist[worst] = d2;
                    bestFraud[worst] = dataset.isFraud(i);
                    worst = indexOfMax(bestDist);
                }
            }
        } finally {
            scanPermits.release();
        }

        int frauds = 0;
        for (boolean f : bestFraud) {
            if (f) {
                frauds++;
            }
        }
        return frauds;
    }

    private static int indexOfMax(long[] values) {
        int max = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[max]) {
                max = i;
            }
        }
        return max;
    }
}
