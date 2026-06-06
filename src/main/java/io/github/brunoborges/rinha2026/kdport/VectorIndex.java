/*
 * Ported from gb/jvmoonshot-xxvi (MIT License), Copyright (c) Gabriel Tobias.
 * Local changes: package/import rewrites for the rinha-2026 offline kdport experiment.
 */
package io.github.brunoborges.rinha2026.kdport;

/**
 * Pluggable nearest-neighbor backend behind {@code FraudScoreHandler}. MoonShot picks one at boot via
 * the {@code INDEX_KIND} env. Lifecycle hooks default to no-ops; only mmap-backed
 * implementations override them.
 */
public interface VectorIndex {

    /**
     * Number of fraud-labeled vectors among the K nearest neighbors; fraud_score = count / K.
     */
    int countFraudsInTopK(float[] query, int k);

    /**
     * Fixed top-5 fast path implementations with hard-coded k=5 internals override this.
     */
    default int countFraudsInTop5(float[] query) {
        return countFraudsInTopK(query, 5);
    }

    /**
     * Exception-free hot-path variant; defaults to {@link #countFraudsInTop5}.
     * {@link io.github.brunoborges.rinha2026.kdport.KdTree} overrides this with a zero-catch implementation.
     */
    default int countFraudsInTop5Fast(float[] query) {
        return countFraudsInTop5(query);
    }

    default int countFraudsInTop5FastI16(short[] query) {
        // Fallback: dequantize I16 to float and delegate.
        float[] fq = new float[16];
        for (int i = 0; i < 14; i++) {
            fq[i] = query[i] * (1.0f / 10000.0f);
        }
        return countFraudsInTop5Fast(fq);
    }

    int size();

    /**
     * Fill {@code out[0..DIMS)} with the feature vector of the node at {@code treeIdx}, in the
     * same permuted lane order stored internally.  Used by {@link com.github.gb.moonshot.WarmupDriver}
     * to sample real index vectors for JIT warmup with realistic input distribution.
     * Default is a no-op; warmup falls back to random vectors for that iteration.
     */
    default void copyNodeVector(int treeIdx, float[] out) {
    }

    /**
     * madvise hints (HUGEPAGE+RANDOM) for mmap'd regions; no-op for heap-resident impls.
     */
    default void applyMmapHints() {
    }

    /**
     * Force every backing page into the page cache so the first request avoids fault tax.
     */
    default void prewarm() {
    }
}
