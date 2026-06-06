/*
 * Ported from gb/jvmoonshot-xxvi (MIT License), Copyright (c) Gabriel Tobias.
 * Local changes: package/import rewrites for the rinha-2026 offline kdport experiment.
 */
package io.github.brunoborges.rinha2026.kdport;

import static io.github.brunoborges.rinha2026.kdport.KdTreeTuning.PROFILING_ENABLED;

/**
 * Read-only probes for the last {@link KdTree} query on the current thread.
 */
public final class KdTreeProbes {

    private static final ThreadLocal<KdTreeScratch> lastScratch = new ThreadLocal<>();

    private KdTreeProbes() {
    }

    static void record(KdTreeScratch scratch) {
        lastScratch.set(scratch);
    }

    private static KdTreeScratch last() {
        KdTreeScratch scratch = lastScratch.get();
        return scratch != null ? scratch : KdTreeScratch.scratchTL.get();
    }

    /**
     * True when {@code KDTREE_PROFILING} env var is set; bench should fail-fast when false.
     */
    public static boolean isProfilingEnabled() {
        return PROFILING_ENABLED;
    }

    public static int lastQueryNodesVisited() {
        return last().visits;
    }

    public static int lastQueryBboxChecks() {
        return last().bboxChecks;
    }

    public static int lastBbfPushes() {
        return last().bbfPushes;
    }

    public static int lastBbfHeapMax() {
        return last().bbfHeapMax;
    }

    public static int lastSlabPrunes() {
        return last().slabPrunes;
    }

    public static int lastBboxPrunes() {
        return last().bboxPrunes;
    }

    public static int lastBboxPrunesLo() {
        return last().bboxPrunesLo;
    }

    public static int lastTopKFilledAt() {
        return last().topKFilledAt;
    }

    public static int lastTopKReplaced() {
        return last().topKReplaced;
    }

    public static float lastFinalPeekDist() {
        return last().finalPeekDist;
    }
}
