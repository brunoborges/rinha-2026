package io.github.brunoborges.rinha2026.kdport;

import io.github.brunoborges.rinha2026.ReferenceDataset;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/** Kill-switch correctness checks for the offline kdport experiment. */
public final class KdPortVerify {
    private static final int DIMS = Dataset.DIMS;
    private static final int STRIDE = Dataset.STRIDE;
    private static final int K = 5;

    public static void main(String[] args) throws Exception {
        requireExactEnv();
        testTinyDeterministic();
        System.out.println("A tiny deterministic trees: PASS");
        testBboxPruneExactRandom();
        System.out.println("B bbox-prune exact==brute force: PASS");
        testTopKTies();
        System.out.println("C top-k ties/duplicates: PASS");
        testQuantizationRoundTrip(Path.of(args.length > 0 ? args[0] : "resources/references.bin"));
        System.out.println("D quantization round-trip: PASS");
        System.out.println("KdPortVerify: PASS");
    }

    private static void requireExactEnv() {
        String[] vars = {
                "KDTREE_MAX_VISITS", "KDTREE_EARLY_DIST_MILLI", "KDTREE_RELAX_EPSILON",
                "KDTREE_RELAX_SOFT_CAP", "KDTREE_RELAX_RANGE", "KDTREE_BUCKET_LEAF_DEPTH",
                "KDTREE_BUCKET_LEAF_MAX_NODES"
        };
        for (String var : vars) {
            if (System.getenv(var) != null) {
                throw new AssertionError("Run verify with exact settings; unset " + var);
            }
        }
    }

    private static void testTinyDeterministic() {
        int[] sizes = {1, 2, 3, 7, 15};
        for (int n : sizes) {
            float[] vecs = new float[n * STRIDE];
            boolean[] labels = new boolean[n];
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < DIMS; d++) {
                    vecs[i * STRIDE + d] = ((i * 997 + d * 37) % 10000) / 10000.0f;
                }
                labels[i] = (i & 1) == 0;
            }
            KdTree tree = KdTreeBuilder.build(new Dataset(n, vecs, labels));
            float[][] queries = {
                    semantic(vecs, 0), semantic(vecs, n - 1), filled(0.0f), filled(0.5f), filled(1.0f)
            };
            for (float[] q : queries) assertMatchesBrute("tiny n=" + n, tree, q);
        }
    }

    private static void testBboxPruneExactRandom() {
        int n = 2000;
        Random rnd = new Random(1234567L);
        float[] vecs = new float[n * STRIDE];
        boolean[] labels = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIMS; d++) {
                int v = rnd.nextInt(10_001);
                vecs[i * STRIDE + d] = v / 10000.0f;
            }
            labels[i] = rnd.nextInt(7) == 0;
        }
        KdTree tree = KdTreeBuilder.build(new Dataset(n, vecs, labels));
        for (int i = 0; i < 2000; i++) {
            float[] q = new float[DIMS];
            for (int d = 0; d < DIMS; d++) {
                int v = rnd.nextInt(10_001);
                q[d] = v / 10000.0f;
            }
            assertMatchesBrute("random query=" + i, tree, q);
        }
    }

    private static void testTopKTies() {
        float[] vecs = new float[12 * STRIDE];
        boolean[] labels = new boolean[12];
        for (int i = 0; i < 12; i++) {
            labels[i] = i == 1 || i == 3 || i == 5 || i == 8;
        }
        KdTree duplicates = KdTreeBuilder.build(new Dataset(12, vecs, labels));
        assertMatchesBrute("all duplicates at origin", duplicates, filled(0.0f));

        Arrays.fill(vecs, 0.0f);
        for (int i = 0; i < 12; i++) {
            vecs[i * STRIDE] = (i & 1) == 0 ? 0.1f : -0.1f;
            labels[i] = (i % 3) == 0;
        }
        KdTree symmetric = KdTreeBuilder.build(new Dataset(12, vecs, labels));
        assertMatchesBrute("symmetric equal-distance", symmetric, filled(0.0f));
    }

    private static void testQuantizationRoundTrip(Path bin) throws Exception {
        if (!Files.exists(bin)) {
            throw new AssertionError("reference dataset not found for round-trip test: " + bin);
        }
        try (ReferenceDataset ds = ReferenceDataset.mmap(bin)) {
            short[] vec16 = ds.toQuantizedArray();
            int samples = Math.min(vec16.length, 100_000);
            for (int i = 0; i < samples; i++) {
                short v = vec16[i];
                short q = KdTree.quantize(v / 10000.0f);
                if (q != v) {
                    throw new AssertionError("round-trip mismatch at sample " + i + ": " + v + " -> " + q);
                }
            }
        }
    }

    private static void assertMatchesBrute(String label, KdTree tree, float[] query) {
        int got = tree.countFraudsInTop5Fast(query);
        int expected = bruteForceTreeOrder(tree, query);
        if (got != expected) {
            throw new AssertionError(label + ": got " + got + ", expected " + expected
                    + ", visits=" + KdTreeProbes.lastQueryNodesVisited());
        }
    }

    private static int bruteForceTreeOrder(KdTree tree, float[] semanticQuery) {
        short[] q = new short[DIMS];
        for (int d = 0; d < DIMS; d++) {
            q[d] = quantizeQueryLikeTree(semanticQuery[KdTree.DIM_PERMUTATION[d]]);
        }
        int[] ids = new int[K];
        int[] sums = new int[K];
        int size = 0;
        for (int treeIdx = 0; treeIdx < tree.n; treeIdx++) {
            int sum = dist(tree, treeIdx, q);
            if (size < K) {
                int pos = size;
                while (pos > 0 && sums[pos - 1] > sum) {
                    sums[pos] = sums[pos - 1];
                    ids[pos] = ids[pos - 1];
                    pos--;
                }
                sums[pos] = sum;
                ids[pos] = treeIdx;
                size++;
            } else if (sum < sums[K - 1]) {
                int pos = K - 1;
                while (pos > 0 && sums[pos - 1] > sum) {
                    sums[pos] = sums[pos - 1];
                    ids[pos] = ids[pos - 1];
                    pos--;
                }
                sums[pos] = sum;
                ids[pos] = treeIdx;
            }
        }
        int frauds = 0;
        for (int i = 0; i < size; i++) frauds += tree.fraud[ids[i]];
        return frauds;
    }

    private static int dist(KdTree tree, int treeIdx, short[] q) {
        int base = treeIdx * KdTree.STRIDE;
        int sum = 0;
        for (int d = 0; d < DIMS; d++) {
            int diff = q[d] - tree.pts[base + d];
            sum += diff * diff;
        }
        return sum;
    }

    private static short quantizeQueryLikeTree(float v) {
        if (v <= -1.0f) return (short) -10_000;
        if (v >= 1.0f) return (short) 10_000;
        return (short) (v * 10_000 + 0.5f);
    }

    private static float[] semantic(float[] vecs, int row) {
        float[] q = new float[DIMS];
        System.arraycopy(vecs, row * STRIDE, q, 0, DIMS);
        return q;
    }

    private static float[] filled(float v) {
        float[] q = new float[DIMS];
        Arrays.fill(q, v);
        return q;
    }

    private KdPortVerify() {
    }
}
