package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * THROWAWAY offline experiment (run on the JVM only; never referenced from
 * {@link App#main}). Builds an implicit balanced KD-tree over the int16 reference
 * vectors and measures, for the contest eval corpus:
 * <ul>
 *   <li>E = FP + FN detection error (exactness vs ground truth),</li>
 *   <li>per-query node-visit distribution p50/p95/p99/max/mean (the go/no-go
 *       metric: does exact KD-tree visit fewer points than the IVF scan?),</li>
 *   <li>agreement with a brute-force exact KNN (catches tie-handling bugs).</li>
 * </ul>
 * Two split strategies are compared: round-robin (dim = depth % 14, no metadata)
 * and max-spread (per-node split dim of largest coordinate range, byte[n]).
 *
 * <pre>java -cp ... KdTreeExperiment [references.bin] [test-data.json]</pre>
 */
public final class KdTreeExperiment {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private static final int K = IvfFraudScorer.K;
    private static final double THRESHOLD = IvfFraudScorer.THRESHOLD;

    // Loaded once.
    static short[] vec16;   // n * DIMS, row-major (original order)
    static byte[] labels;   // n
    static int n;

    // Tree (treePos -> original index).
    static int[] order;
    static byte[] splitDim; // per treePos; null for round-robin
    static boolean useStoredDim;

    public static void main(String[] args) throws Exception {
        Path bin = Path.of(args.length > 0 ? args[0] : "resources/references.bin");
        Path data = Path.of(args.length > 1 ? args[1] : "test/test-data.json");

        TransactionVectorizer vectorizer = new TransactionVectorizer();
        ReferenceDataset ds = ReferenceDataset.mmap(bin);
        ds.preload();
        vec16 = ds.toQuantizedArray();
        labels = ds.toLabelArray();
        n = ds.count();
        flushf("== %s : %,d vectors ==%n", bin.getFileName(), n);

        List<double[]> qvecs = new ArrayList<>();
        List<Boolean> expected = new ArrayList<>();
        loadCorpus(data, vectorizer, qvecs, expected);
        int q = qvecs.size();
        short[][] q16 = new short[q][];
        for (int i = 0; i < q; i++) {
            short[] qq = new short[DIMS];
            ReferenceDataset.quantizeInto(qvecs.get(i), qq);
            q16[i] = qq;
        }
        flushf("corpus: %,d queries%n", q);

        // Brute-force gold on a sample (full 54k x 3M is ~40 min single-thread): parallel, exact
        // top-5 fraud count per query (lexicographic (dist,idx) tie-break). Used to (a) report the
        // exact E floor on the sample and (b) verify KD-tree agreement on the same sample.
        int sampleSize = Math.min(q, Integer.getInteger("sample", 4000));
        int[] sampleIdx = new int[sampleSize];
        java.util.Random rnd = new java.util.Random(42);
        for (int i = 0; i < sampleSize; i++) sampleIdx[i] = rnd.nextInt(q);
        int[] goldSample = new int[sampleSize];
        long bfStart = System.nanoTime();
        java.util.stream.IntStream.range(0, sampleSize).parallel()
                .forEach(i -> goldSample[i] = bruteForceFrauds(q16[sampleIdx[i]]));
        int sampleFp = 0, sampleFn = 0;
        for (int i = 0; i < sampleSize; i++) {
            boolean approved = (double) goldSample[i] / K < THRESHOLD;
            boolean expFraud = !expected.get(sampleIdx[i]);
            if (expFraud) { if (approved) sampleFn++; } else { if (!approved) sampleFp++; }
        }
        flushf("brute-force gold (%,d sampled) in %,d ms; sample E=%d (FP=%d FN=%d)%n",
                sampleSize, (System.nanoTime() - bfStart) / 1_000_000, sampleFp + sampleFn, sampleFp, sampleFn);

        buildTree(true); // max-spread split (stored per-node dim)
        flushf("budget    E   vs-gold-mism   visits:p50    p95     p99     max    mean   ms/q%n");
        for (int budget : new int[] {500, 1000, 2000, 4000, 8000, 16000, 32000}) {
            runBbf(budget, q16, expected, goldSample, sampleIdx);
        }
        ds.close();
    }

    // ---- tree build (in-place median partition over an index permutation) -----------------------

    static void buildTree(boolean storedDim) {
        useStoredDim = storedDim;
        order = new int[n];
        for (int i = 0; i < n; i++) order[i] = i;
        splitDim = storedDim ? new byte[n] : null;
        long t = System.nanoTime();
        build(0, n, 0);
        flushf("[%s] built in %,d ms%n",
                storedDim ? "max-spread" : "round-robin", (System.nanoTime() - t) / 1_000_000);
    }

    static void build(int lo, int hi, int depth) {
        if (hi - lo <= 1) {
            if (hi - lo == 1 && useStoredDim) splitDim[lo] = 0;
            return;
        }
        int m = (lo + hi) >>> 1;
        int dim = useStoredDim ? widestDim(lo, hi) : depth % DIMS;
        nthElement(lo, hi, m, dim);
        if (useStoredDim) splitDim[m] = (byte) dim;
        build(lo, m, depth + 1);
        build(m + 1, hi, depth + 1);
    }

    static int widestDim(int lo, int hi) {
        int bestDim = 0;
        long bestRange = -1;
        for (int d = 0; d < DIMS; d++) {
            short mn = Short.MAX_VALUE, mx = Short.MIN_VALUE;
            for (int i = lo; i < hi; i++) {
                short v = vec16[order[i] * DIMS + d];
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            long r = mx - mn;
            if (r > bestRange) { bestRange = r; bestDim = d; }
        }
        return bestDim;
    }

    /** Quickselect: place the k-th smallest (by coord on dim, tie-break original index) at order[k]. */
    static void nthElement(int lo, int hi, int k, int dim) {
        int l = lo, h = hi - 1;
        while (l < h) {
            int pivot = order[l + ((h - l) >>> 1)];
            short pv = vec16[pivot * DIMS + dim];
            int i = l, j = h;
            while (i <= j) {
                while (cmp(order[i], pv, pivot, dim) < 0) i++;
                while (cmp(order[j], pv, pivot, dim) > 0) j--;
                if (i <= j) { int tmp = order[i]; order[i] = order[j]; order[j] = tmp; i++; j--; }
            }
            if (k <= j) h = j;
            else if (k >= i) l = i;
            else break;
        }
    }

    /** Compare reference idx's coord on dim to pivot value; tie-break by original index. */
    static int cmp(int idx, short pv, int pivotIdx, int dim) {
        short v = vec16[idx * DIMS + dim];
        if (v != pv) return Integer.compare(v, pv);
        return Integer.compare(idx, pivotIdx);
    }

    // ---- bounded BBF (best-bin-first) KD-tree search --------------------------------------------

    static final int LEAF = 32; // ranges this small are scanned linearly

    static long[] bestDist = new long[K];
    static int[] bestIdx = new int[K];
    static int bestCount;
    static int worstSlot;
    static long visits;

    // Min-heap of pending subtrees keyed by lower-bound distance. Parallel arrays.
    static long[] hLb = new long[1 << 18];
    static int[] hLo = new int[1 << 18];
    static int[] hHi = new int[1 << 18];
    static int hSize;

    static int searchBbf(short[] qq, int budget) {
        Arrays.fill(bestDist, Long.MAX_VALUE);
        Arrays.fill(bestIdx, Integer.MAX_VALUE);
        bestCount = 0;
        worstSlot = 0;
        visits = 0;
        hSize = 0;
        heapPush(0L, 0, n);
        while (hSize > 0 && visits < budget) {
            long lb = hLb[0];
            int lo = hLo[0], hi = hHi[0];
            heapPop();
            // Prune: if the closest possible point in this subtree is farther than our current
            // 5th-best, nothing here can improve the result.
            if (bestCount == K && lb > bestDist[worstSlot]) continue;
            if (hi - lo <= LEAF) {
                for (int i = lo; i < hi; i++) {
                    int oi = order[i];
                    consider(dist(qq, oi), oi);
                }
                visits += (hi - lo);
                continue;
            }
            int m = (lo + hi) >>> 1;
            int oi = order[m];
            consider(dist(qq, oi), oi);
            visits++;
            int dim = splitDim[m] & 0xFF;
            int diff = qq[dim] - vec16[oi * DIMS + dim];
            long axis = (long) diff * diff;
            if (diff <= 0) {
                heapPush(lb, lo, m);          // near side: same lower bound
                heapPush(lb + axis, m + 1, hi); // far side: must cross the split plane
            } else {
                heapPush(lb, m + 1, hi);
                heapPush(lb + axis, lo, m);
            }
        }
        int f = 0;
        for (int i = 0; i < K; i++) if (bestIdx[i] != Integer.MAX_VALUE && labels[bestIdx[i]] != 0) f++;
        return f;
    }

    static void heapPush(long lb, int lo, int hi) {
        if (lo >= hi) return;
        if (hSize >= hLb.length) return; // experiment-only guard; budgets keep this well bounded
        int i = hSize++;
        hLb[i] = lb; hLo[i] = lo; hHi[i] = hi;
        while (i > 0) {
            int p = (i - 1) >>> 1;
            if (hLb[p] <= hLb[i]) break;
            swapHeap(i, p); i = p;
        }
    }

    static void heapPop() {
        int last = --hSize;
        hLb[0] = hLb[last]; hLo[0] = hLo[last]; hHi[0] = hHi[last];
        int i = 0;
        while (true) {
            int l = 2 * i + 1, r = l + 1, s = i;
            if (l < hSize && hLb[l] < hLb[s]) s = l;
            if (r < hSize && hLb[r] < hLb[s]) s = r;
            if (s == i) break;
            swapHeap(i, s); i = s;
        }
    }

    static void swapHeap(int a, int b) {
        long tl = hLb[a]; hLb[a] = hLb[b]; hLb[b] = tl;
        int t1 = hLo[a]; hLo[a] = hLo[b]; hLo[b] = t1;
        int t2 = hHi[a]; hHi[a] = hHi[b]; hHi[b] = t2;
    }

    /** Insert (d, idx) into the size-K best set, lexicographic (dist, idx). */
    static void consider(long d, int idx) {
        if (bestCount < K) {
            bestDist[bestCount] = d;
            bestIdx[bestCount] = idx;
            bestCount++;
            if (bestCount == K) worstSlot = lexMax();
            return;
        }
        if (d < bestDist[worstSlot] || (d == bestDist[worstSlot] && idx < bestIdx[worstSlot])) {
            bestDist[worstSlot] = d;
            bestIdx[worstSlot] = idx;
            worstSlot = lexMax();
        }
    }

    static int lexMax() {
        int w = 0;
        for (int i = 1; i < K; i++) {
            if (bestDist[i] > bestDist[w] || (bestDist[i] == bestDist[w] && bestIdx[i] > bestIdx[w])) w = i;
        }
        return w;
    }

    // ---- brute-force gold -----------------------------------------------------------------------

    static int bruteForceFrauds(short[] qq) {
        long[] bd = new long[K];
        int[] bi = new int[K];
        Arrays.fill(bd, Long.MAX_VALUE);
        Arrays.fill(bi, Integer.MAX_VALUE);
        int cnt = 0, worst = 0;
        for (int idx = 0; idx < n; idx++) {
            long d = dist(qq, idx);
            if (cnt < K) {
                bd[cnt] = d; bi[cnt] = idx; cnt++;
                if (cnt == K) worst = lexMaxOf(bd, bi);
            } else if (d < bd[worst] || (d == bd[worst] && idx < bi[worst])) {
                bd[worst] = d; bi[worst] = idx; worst = lexMaxOf(bd, bi);
            }
        }
        int f = 0;
        for (int i = 0; i < K; i++) if (bi[i] != Integer.MAX_VALUE && labels[bi[i]] != 0) f++;
        return f;
    }

    static int lexMaxOf(long[] bd, int[] bi) {
        int w = 0;
        for (int i = 1; i < K; i++) {
            if (bd[i] > bd[w] || (bd[i] == bd[w] && bi[i] > bi[w])) w = i;
        }
        return w;
    }

    static long dist(short[] qq, int idx) {
        long s = 0;
        int off = idx * DIMS;
        for (int d = 0; d < DIMS; d++) { int a = qq[d] - vec16[off + d]; s += (long) a * a; }
        return s;
    }

    // ---- run + report ---------------------------------------------------------------------------

    static void runBbf(int budget, short[][] q16, List<Boolean> expected,
                       int[] goldSample, int[] sampleIdx) {
        int q = q16.length;
        long[] vis = new long[q];
        int[] frauds = new int[q];
        long t = System.nanoTime();
        for (int i = 0; i < q; i++) {
            frauds[i] = searchBbf(q16[i], budget);
            vis[i] = visits;
        }
        long ms = (System.nanoTime() - t) / 1_000_000;
        int mismatch = 0;
        for (int i = 0; i < sampleIdx.length; i++) {
            if (searchBbf(q16[sampleIdx[i]], budget) != goldSample[i]) mismatch++;
        }
        int e = errorOf(frauds, expected);
        long[] sorted = vis.clone();
        Arrays.sort(sorted);
        long sum = 0; for (long v : sorted) sum += v;
        flushf("%-8d  %3d   %6d      %,8d %,7d %,7d %,7d %,7d   %.3f%n",
                budget, e, mismatch,
                pct(sorted, 50), pct(sorted, 95), pct(sorted, 99), sorted[q - 1], sum / q,
                (double) ms / q);
    }

    static int errorOf(int[] frauds, List<Boolean> expected) {
        int fp = 0, fn = 0;
        for (int i = 0; i < frauds.length; i++) {
            boolean approved = (double) frauds[i] / K < THRESHOLD;
            boolean expFraud = !expected.get(i);
            if (expFraud) { if (approved) fn++; } else { if (!approved) fp++; }
        }
        return fp + fn;
    }

    static long pct(long[] sorted, int p) {
        int idx = (int) Math.min(sorted.length - 1L, (long) Math.ceil(p / 100.0 * sorted.length) - 1);
        return sorted[Math.max(0, idx)];
    }

    static void loadCorpus(Path data, TransactionVectorizer vectorizer,
                           List<double[]> qvecs, List<Boolean> expected) throws Exception {
        JsonFactory factory = new JsonFactory();
        RequestVectorParser parser = new RequestVectorParser(vectorizer);
        try (JsonParser p = factory.createParser(Files.newInputStream(data))) {
            p.nextToken();
            while (p.nextToken() != JsonToken.END_OBJECT) {
                if (!"entries".equals(p.currentName())) { p.nextToken(); p.skipChildren(); continue; }
                p.nextToken();
                while (p.nextToken() != JsonToken.END_ARRAY) {
                    byte[] reqBytes = null; Boolean approved = null;
                    while (p.nextToken() != JsonToken.END_OBJECT) {
                        switch (p.currentName()) {
                            case "request" -> {
                                p.nextToken();
                                ByteArrayOutputStream buf = new ByteArrayOutputStream(512);
                                try (JsonGenerator g = factory.createGenerator(buf)) { g.copyCurrentStructure(p); }
                                reqBytes = buf.toByteArray();
                            }
                            case "expected_approved" -> approved = p.nextBooleanValue();
                            default -> { p.nextToken(); p.skipChildren(); }
                        }
                    }
                    RequestVectorParser.State st = parser.acquire();
                    try {
                        parser.vectorize(new ByteArrayInputStream(reqBytes), st);
                        qvecs.add(Arrays.copyOf(st.qvec, DIMS));
                        expected.add(approved);
                    } finally {
                        parser.release(st);
                    }
                }
            }
        }
    }

    private KdTreeExperiment() {
    }

    static void flushf(String fmt, Object... args) {
        System.out.printf(fmt, args);
        System.out.flush();
    }
}
