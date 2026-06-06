package io.github.brunoborges.rinha2026;

import io.github.brunoborges.rinha2026.kdport.KdTree;

/**
 * Exact k-NN fraud scorer backed by the ported {@link KdTree} index (bbox-pruned
 * best-bin-first over i16-quantized reference vectors, mmap'd off-heap).
 *
 * <p>Replaces the approximate IVF scan on the hot path: an offline experiment over
 * the 54,100-query eval corpus measured E=0 (FP=0, FN=0) at ~2,900 p99 node visits
 * — perfect detection at roughly an order of magnitude fewer candidate evaluations
 * than the IVF (~27k for single-digit E). IVF is retained as a fallback (selectable
 * via {@code SCORER=ivf}).
 *
 * <p>The data path is the single {@code rinha-epoll} thread, so the tree's owned
 * {@code instanceScratch} (one per instance) and the reusable query buffer here are
 * allocation-free and contention-free. {@link KdTree#countFraudsInTop5Fast(float[])}
 * quantizes and permutes the query internally with the exact build-time transform,
 * matching the validated offline path bit-for-bit.
 */
public final class KdTreeFraudScorer implements FraudScorer {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;

    private final TransactionVectorizer vectorizer;
    private final KdTree tree;

    // Confined to the single epoll thread (see class doc); reused per request to stay alloc-free.
    private final float[] queryBuf = new float[DIMS];
    private final double[] vecBuf = new double[DIMS];

    public KdTreeFraudScorer(TransactionVectorizer vectorizer, KdTree tree) {
        this.vectorizer = vectorizer;
        this.tree = tree;
    }

    @Override
    public FraudResponse score(FraudRequest request) {
        vectorizer.vectorizeInto(request, vecBuf);
        return scoreVector(vecBuf);
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
        float[] q = queryBuf;
        for (int d = 0; d < DIMS; d++) {
            q[d] = (float) qvec[d];
        }
        int frauds = tree.countFraudsInTop5Fast(q);
        double fraudScore = (double) frauds / IvfFraudScorer.K;
        boolean approved = fraudScore < IvfFraudScorer.THRESHOLD;
        return new FraudResponse(approved, fraudScore);
    }

    @Override
    public void preload() {
        tree.applyMmapHints();
        tree.prewarm();
    }
}
