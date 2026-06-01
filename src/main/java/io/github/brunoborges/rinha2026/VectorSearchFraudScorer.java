package io.github.brunoborges.rinha2026;

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

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
 * scoring is bounded by a pool of reusable {@link Scratch} buffers sized by the
 * {@code WORKERS} knob (see {@link Concurrency}); HTTP concurrency (virtual
 * threads) is unaffected. The
 * pool doubles as the allocation arena, so steady-state scoring allocates
 * nothing on the heap.
 */
public final class VectorSearchFraudScorer implements FraudScorer {

    /** Number of nearest neighbors considered. */
    public static final int K = 5;

    /** Approval threshold: approved when {@code fraud_score < THRESHOLD}. */
    public static final double THRESHOLD = 0.6;

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;

    private final TransactionVectorizer vectorizer;
    private final ReferenceDataset dataset;
    private final ArrayBlockingQueue<Scratch> scratchPool;

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
        int permits = Concurrency.workers();
        this.scratchPool = new ArrayBlockingQueue<>(permits);
        for (int i = 0; i < permits; i++) {
            this.scratchPool.add(new Scratch());
        }
    }

    @Override
    public FraudResponse score(FraudRequest request) {
        int frauds;
        Scratch s = borrow();
        try {
            vectorizer.vectorizeInto(request, s.qvec);
            ReferenceDataset.quantizeInto(s.qvec, s.query);
            frauds = countFraudsAmongNearest(s);
        } finally {
            scratchPool.offer(s);
        }
        return toResponse(frauds);
    }

    @Override
    public boolean supportsVectorInput() {
        return true;
    }

    @Override
    public TransactionVectorizer vectorizer() {
        return vectorizer;
    }

    @Override
    public FraudResponse scoreVector(double[] qvec) {
        int frauds;
        Scratch s = borrow();
        try {
            ReferenceDataset.quantizeInto(qvec, s.query);
            frauds = countFraudsAmongNearest(s);
        } finally {
            scratchPool.offer(s);
        }
        return toResponse(frauds);
    }

    private FraudResponse toResponse(int frauds) {
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
     * Finds the {@code min(K, count)} nearest references to {@code s.query} and
     * returns how many are labeled fraud. Maintains a fixed-size buffer of the
     * smallest squared distances seen so far; ties keep the earlier record
     * (strict {@code <} replacement). The neighbor buffers in {@code s} are reset
     * over their active prefix so a reused Scratch never leaks prior results.
     */
    private int countFraudsAmongNearest(Scratch s) {
        int count = dataset.count();
        int k = Math.min(K, count);
        short[] query = s.query;

        long[] bestDist = s.bestDist;
        boolean[] bestFraud = s.bestFraud;
        Arrays.fill(bestDist, 0, k, Long.MAX_VALUE);
        Arrays.fill(bestFraud, 0, k, false);
        int worst = 0; // index of the current largest distance in bestDist

        for (int i = 0; i < count; i++) {
            long d2 = dataset.squaredDistance(query, i);
            if (d2 < bestDist[worst]) {
                bestDist[worst] = d2;
                bestFraud[worst] = dataset.isFraud(i);
                worst = indexOfMax(bestDist, k);
            }
        }

        int frauds = 0;
        for (int i = 0; i < k; i++) {
            if (bestFraud[i]) {
                frauds++;
            }
        }
        return frauds;
    }

    /** Takes a Scratch from the pool, retrying through interrupts. */
    private Scratch borrow() {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    return scratchPool.take();
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    private static int indexOfMax(long[] values, int len) {
        int max = 0;
        for (int i = 1; i < len; i++) {
            if (values[i] > values[max]) {
                max = i;
            }
        }
        return max;
    }

    /** Per-request working memory, recycled through {@link #scratchPool}. */
    private static final class Scratch {
        final double[] qvec = new double[DIMS];
        final short[] query = new short[DIMS];
        final long[] bestDist = new long[K];
        final boolean[] bestFraud = new boolean[K];
    }
}
