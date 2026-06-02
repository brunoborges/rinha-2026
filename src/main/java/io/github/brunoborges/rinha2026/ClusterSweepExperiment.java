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
 * THROWAWAY offline experiment: for a given IVF index ({@code references.bin} built
 * at some cluster count), sweep NPROBE (fixed and adaptive) and report detection
 * error E=FP+FN AND the per-query candidate-count distribution. Answers whether a
 * finer index lets us scan materially fewer candidates while keeping E=9 (exact
 * int16 distances — the only lossless lever after int8 was shown to wreck E).
 *
 * <p>Run on the JVM only; never referenced from {@link App#main}. Delete when done.
 *
 * <pre>java -cp ... ClusterSweepExperiment &lt;references.bin&gt; [test-data.json]</pre>
 */
public final class ClusterSweepExperiment {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private static final int K = IvfFraudScorer.K;
    private static final double THRESHOLD = IvfFraudScorer.THRESHOLD;
    private static final int SCAN_CAP = IvfFraudScorer.DEFAULT_SCAN_CAP;
    private static final int REFINE_MIN = IvfFraudScorer.REFINE_MIN_FRAUDS;
    private static final int REFINE_MAX = IvfFraudScorer.REFINE_MAX_FRAUDS;

    public static void main(String[] args) throws Exception {
        Path bin = Path.of(args.length > 0 ? args[0] : "resources/references.bin");
        Path data = Path.of(args.length > 1 ? args[1] : "test/test-data.json");

        TransactionVectorizer vectorizer = new TransactionVectorizer();
        ReferenceDataset ds = ReferenceDataset.mmap(bin);
        ds.preload();
        short[] vec16 = ds.toQuantizedArray();
        byte[] labels = ds.toLabelArray();
        int clusters = ds.clusters();
        System.out.printf("== %s : %,d vectors, %d clusters ==%n", bin.getFileName(), ds.count(), clusters);

        List<double[]> qvecs = new ArrayList<>();
        List<Boolean> expected = new ArrayList<>();
        loadCorpus(data, vectorizer, qvecs, expected);
        int n = qvecs.size();
        short[][] q16 = new short[n][];
        for (int i = 0; i < n; i++) {
            short[] q = new short[DIMS];
            ReferenceDataset.quantizeInto(qvecs.get(i), q);
            q16[i] = q;
        }

        System.out.println("nprobe(fixed)  E  FP FN   cand:p50   p95     p99     max     mean");
        for (int np : new int[] {2, 4, 6, 8, 12, 16, 24}) {
            if (np > clusters) continue;
            run(ds, vec16, labels, q16, expected, np, np);
        }
        System.out.println("nprobe(adapt)->24");
        for (int base : new int[] {2, 4, 6, 8}) {
            if (base > clusters) continue;
            run(ds, vec16, labels, q16, expected, base, Math.min(24, clusters));
        }
        ds.close();
    }

    private static void run(ReferenceDataset ds, short[] vec16, byte[] labels, short[][] q16,
                            List<Boolean> expected, int basep, int maxp) {
        int n = q16.length;
        long[] cand = new long[n];
        int fp = 0, fn = 0;
        int[] cid = new int[maxp];
        long[] cdist = new long[maxp];
        long[] bestDist = new long[K];
        boolean[] bestFraud = new boolean[K];
        for (int i = 0; i < n; i++) {
            int[] r = scan(ds, vec16, labels, q16[i], basep, maxp, cid, cdist, bestDist, bestFraud);
            int frauds = r[0];
            cand[i] = r[1];
            boolean approved = (double) frauds / K < THRESHOLD;
            boolean expFraud = !expected.get(i);
            if (expFraud) { if (approved) fn++; } else { if (!approved) fp++; }
        }
        Arrays.sort(cand);
        long sum = 0; for (long c : cand) sum += c;
        String label = (basep == maxp) ? String.format("%6d", basep)
                : String.format("%4d->%d", basep, maxp);
        System.out.printf("%-9s  %3d %3d %3d   %7d %7d %7d %7d %8d%n",
                label, fp + fn, fp, fn,
                pct(cand, 50), pct(cand, 95), pct(cand, 99), cand[n - 1], sum / n);
    }

    private static long pct(long[] sorted, int p) {
        int idx = (int) Math.min(sorted.length - 1L, (long) Math.ceil(p / 100.0 * sorted.length) - 1);
        return sorted[Math.max(0, idx)];
    }

    /** Returns {frauds, candidatesScanned}. Adaptive when maxp>basep. */
    private static int[] scan(ReferenceDataset ds, short[] vec16, byte[] labels, short[] query,
                              int basep, int maxp, int[] cid, long[] cdist,
                              long[] bestDist, boolean[] bestFraud) {
        int probed = selectClusters(ds, query, cid, cdist, maxp);
        int base = Math.min(basep, probed);
        Arrays.fill(bestDist, Long.MAX_VALUE);
        Arrays.fill(bestFraud, false);
        int[] cur = {0, 0}; // worstNeighbor, scanned
        scanClusters(ds, vec16, labels, query, cid, 0, base, bestDist, bestFraud, cur);
        int frauds = count(bestFraud);
        if (probed > base && frauds >= REFINE_MIN && frauds <= REFINE_MAX) {
            scanClusters(ds, vec16, labels, query, cid, base, probed, bestDist, bestFraud, cur);
            frauds = count(bestFraud);
        }
        return new int[] {frauds, cur[1]};
    }

    private static int selectClusters(ReferenceDataset ds, short[] query, int[] cid, long[] cdist, int maxp) {
        int clusters = ds.clusters();
        maxp = Math.min(maxp, clusters);
        Arrays.fill(cdist, 0, maxp, Long.MAX_VALUE);
        int worst = 0;
        for (int c = 0; c < clusters; c++) {
            long d = ds.centroidSquaredDistance(query, c);
            if (d < cdist[worst]) {
                cdist[worst] = d; cid[worst] = c;
                worst = indexOfMax(cdist, maxp);
            }
        }
        for (int i = 1; i < maxp; i++) {
            long d = cdist[i]; int c = cid[i]; int j = i - 1;
            while (j >= 0 && cdist[j] > d) { cdist[j+1]=cdist[j]; cid[j+1]=cid[j]; j--; }
            cdist[j+1]=d; cid[j+1]=c;
        }
        return maxp;
    }

    private static void scanClusters(ReferenceDataset ds, short[] vec16, byte[] labels, short[] query,
                                     int[] cid, int from, int to, long[] bestDist, boolean[] bestFraud,
                                     int[] cur) {
        int worst = cur[0], scanned = cur[1];
        outer:
        for (int j = from; j < to; j++) {
            int c = cid[j];
            int start = ds.clusterStart(c), end = ds.clusterEnd(c);
            for (int idx = start; idx < end; idx++) {
                if (scanned >= SCAN_CAP) break outer;
                long d2 = dist(query, vec16, idx * DIMS);
                if (d2 < bestDist[worst]) {
                    bestDist[worst] = d2;
                    bestFraud[worst] = labels[idx] != 0;
                    worst = indexOfMax(bestDist, K);
                }
                scanned++;
            }
        }
        cur[0] = worst; cur[1] = scanned;
    }

    private static long dist(short[] query, short[] buf, int off) {
        long s = 0;
        for (int d = 0; d < DIMS; d++) { int a = query[d] - buf[off + d]; s += (long) a * a; }
        return s;
    }

    private static int count(boolean[] f) { int c = 0; for (boolean b : f) if (b) c++; return c; }

    private static int indexOfMax(long[] v, int len) {
        int m = 0; for (int i = 1; i < len; i++) if (v[i] > v[m]) m = i; return m;
    }

    private static void loadCorpus(Path data, TransactionVectorizer vectorizer,
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
}
