package io.github.brunoborges.rinha2026;

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.logging.Logger;

/**
 * {@link FraudScorer} backed by an IVF (inverted-file) approximate
 * nearest-neighbor search over an indexed {@link ReferenceDataset} (format
 * version 2). It replaces the brute-force full scan of
 * {@link VectorSearchFraudScorer} so the service can sustain the challenge's
 * throughput on a single CPU.
 *
 * <p>For each request the query is vectorized and quantized; the
 * {@code nprobe} nearest cluster centroids are found, and only the records in
 * those clusters are scanned for the {@value #K} nearest neighbors. The fraud
 * score is the fraction of those neighbors labeled fraud, approved when below
 * {@value #THRESHOLD} &mdash; identical decision rule to the brute-force scorer,
 * but over a small candidate set.
 *
 * <p>Probed clusters are scanned in ascending centroid-distance order and a
 * {@code scanCap} bounds the total records examined, so a query whose probed
 * clusters happen to be unusually large still has bounded tail latency.
 */
public final class IvfFraudScorer implements FraudScorer {

    private static final Logger LOG = Logger.getLogger(IvfFraudScorer.class.getName());

    /** Number of nearest neighbors considered. */
    public static final int K = 5;

    /** Approval threshold: approved when {@code fraud_score < THRESHOLD}. */
    public static final double THRESHOLD = 0.6;

    /**
     * Baseline number of clusters probed per query unless overridden by
     * {@code NPROBE}. Almost every query is decided at this probe count.
     *
     * <p>Tuned empirically against the official load test (see docs/EVALUATION.md
     * scoring). The k=5 decision is unambiguous for the vast majority of requests
     * (the fraud-count among the 5 nearest neighbors lands at 0/1 or 4/5), and for
     * those NPROBE=6 already matches an exhaustive scan. Probing more clusters only
     * helps the small fraction of <em>boundary</em> queries (fraud-count 2&ndash;4),
     * which is exactly what {@link #DEFAULT_MAX_NPROBE adaptive refinement} targets.
     * Offline validation over the 54,100-row eval set shows NPROBE=6 holds the exact
     * same error floor as NPROBE=8 (E=19, FP=4/FN=5) while scanning ~23% fewer
     * vectors per query on average (mean clusters 6.56 vs 8.50; refinement rate is
     * unchanged at ~3.1%) &mdash; the smaller baseline scan adds throughput headroom
     * near CPU saturation, which lowers the queuing-dominated p99 tail. NPROBE=5
     * regresses detection (E=22), so 6 is the floor.
     */
    public static final int DEFAULT_NPROBE = 6;

    /**
     * Maximum clusters probed when a query is <em>adaptively refined</em>, unless
     * overridden by {@code MAX_NPROBE}.
     *
     * <p>All observed detection errors live at the decision boundary: a query whose
     * baseline fraud-count is {@value #REFINE_MIN_FRAUDS}&ndash;{@value #REFINE_MAX_FRAUDS}
     * of {@value #K} is one neighbor away from flipping its approve/decline verdict.
     * Only those queries (~3% of traffic) continue scanning out to {@code MAX_NPROBE}
     * clusters; the extra candidates refine the k=5 neighborhood and correct most
     * boundary misclassifications. Because the centroid shortlist is already computed
     * out to {@code MAX_NPROBE} and the running top-k heap is simply extended, the
     * refined result is identical to a uniform {@code MAX_NPROBE} scan &mdash; but the
     * cost is paid on only the boundary queries, so average scan work (and p99) stay
     * close to the {@code NPROBE}=6 baseline while detection improves markedly.
     *
     * <p>Default lowered from 24 to 16 after tuning against the <em>contest score
     * formula</em> (not detection alone): the score's p99 term is steep
     * ({@code 1000·log10(1000/p99)}) while the detection penalty is shallow
     * ({@code -300·log10(1+E)}) and its rate component is clamped for E&le;54. Capping
     * the adaptive ramp at 16 trims the worst-case scan tail (E rises only 19&rarr;26)
     * and on the VM lifted the mean final_score from ~4433 (MAX=24) to ~4788.
     */
    public static final int DEFAULT_MAX_NPROBE = 16;

    /** Inclusive lower bound of the baseline fraud-count band that triggers refinement. */
    public static final int REFINE_MIN_FRAUDS = 2;

    /** Inclusive upper bound of the baseline fraud-count band that triggers refinement. */
    public static final int REFINE_MAX_FRAUDS = 4;

    /** Maximum records scanned per query unless overridden by {@code SCAN_CAP}. */
    public static final int DEFAULT_SCAN_CAP = 120_000;

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;

    /**
     * Number of vectors copied off-heap into a heap buffer per bulk
     * {@link ReferenceDataset#copyVectorRange} call. Bounds the per-scan buffer
     * to {@code CHUNK * DIMENSIONS} shorts (~224&nbsp;KiB) while staying large
     * enough that most clusters copy in one shot.
     */
    private static final int CHUNK = 8192;

    private final TransactionVectorizer vectorizer;
    private final ReferenceDataset dataset;
    private final ArrayBlockingQueue<Scratch> scratchPool;
    private final int baseNprobe;
    private final int maxNprobe;
    private final int scanCap;

    public IvfFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset) {
        this(vectorizer, dataset, resolveInt("NPROBE", DEFAULT_NPROBE),
                resolveInt("MAX_NPROBE", DEFAULT_MAX_NPROBE),
                resolveInt("SCAN_CAP", DEFAULT_SCAN_CAP));
    }

    /**
     * Fixed-probe constructor (no adaptive refinement): every query scans exactly
     * {@code nprobe} clusters. Retained for tests and offline harnesses that need
     * deterministic uniform behavior.
     */
    public IvfFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset,
                          int nprobe, int scanCap) {
        this(vectorizer, dataset, nprobe, nprobe, scanCap);
    }

    public IvfFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset,
                          int baseNprobe, int maxNprobe, int scanCap) {
        if (!dataset.hasIndex()) {
            throw new IllegalArgumentException("dataset has no IVF index; use VectorSearchFraudScorer");
        }
        this.vectorizer = vectorizer;
        this.dataset = dataset;
        int clusters = dataset.clusters();
        this.baseNprobe = Math.max(1, Math.min(baseNprobe, clusters));
        this.maxNprobe = Math.max(this.baseNprobe, Math.min(maxNprobe, clusters));
        this.scanCap = Math.max(K, scanCap);
        // One reusable Scratch per permit; the bounded pool both caps concurrent
        // scans (take() blocks when empty) and recycles every per-request buffer,
        // so scoring allocates nothing on the heap after warmup. Sized by WORKERS
        // (not availableProcessors, which the cgroup pins to 1) so several
        // memory-stall-bound scans can overlap within the CPU quota.
        int permits = Concurrency.workers();
        this.scratchPool = new ArrayBlockingQueue<>(permits);
        for (int i = 0; i < permits; i++) {
            this.scratchPool.add(new Scratch(this.maxNprobe));
        }
        LOG.info(() -> "IVF scorer: clusters=" + dataset.clusters() + ", nprobe=" + this.baseNprobe
                + ", maxNprobe=" + this.maxNprobe + ", refineBand=[" + REFINE_MIN_FRAUDS + ","
                + REFINE_MAX_FRAUDS + "], scanCap=" + this.scanCap + ", scanThreads=" + permits);
    }

    @Override
    public FraudResponse score(FraudRequest request) {
        int frauds;
        Scratch s = borrow();
        try {
            vectorizer.vectorizeInto(request, s.qvec);
            ReferenceDataset.quantizeInto(s.qvec, s.query);
            frauds = scan(s.query, s);
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
            frauds = scan(s.query, s);
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
     * Package-private hook used by tests to exercise the IVF candidate search
     * directly with a quantized query, bypassing vectorization. Borrows a pooled
     * {@link Scratch} so it shares the production allocation-free scan path.
     */
    int countFraudsAmongNearest(short[] query) {
        Scratch s = borrow();
        try {
            return scan(query, s);
        } finally {
            scratchPool.offer(s);
        }
    }

    /**
     * Core IVF search over {@code query} using the caller-owned {@code s} for all
     * working memory (centroid shortlist, neighbor heap, and the bulk-copy
     * buffer). Every reused region is reset over its active prefix before use, so
     * no state leaks between requests.
     */
    private int scan(short[] query, Scratch s) {
        int clusters = dataset.clusters();
        int maxp = Math.min(maxNprobe, clusters);
        int basep = Math.min(baseNprobe, clusters);

        // Find the maxp nearest centroids (bounded buffer of (distance, clusterId)).
        // The shortlist is always built out to maxp so adaptive refinement can extend
        // the candidate scan without recomputing the (fixed O(clusters)) centroid pass.
        long[] cdist = s.cdist;
        int[] cid = s.cid;
        Arrays.fill(cdist, 0, maxp, Long.MAX_VALUE);
        int worst = 0;
        for (int c = 0; c < clusters; c++) {
            long d = dataset.centroidSquaredDistance(query, c);
            if (d < cdist[worst]) {
                cdist[worst] = d;
                cid[worst] = c;
                worst = indexOfMax(cdist, maxp);
            }
        }
        sortByDistance(cdist, cid, maxp);

        int count = dataset.count();
        int k = Math.min(K, count);
        long[] bestDist = s.bestDist;
        boolean[] bestFraud = s.bestFraud;
        Arrays.fill(bestDist, 0, k, Long.MAX_VALUE);
        Arrays.fill(bestFraud, 0, k, false);

        s.worstNeighbor = 0;
        s.scanned = 0;

        // Baseline pass: scan the basep nearest clusters into the top-k heap.
        scanClusters(query, s, cid, 0, basep, k);
        int frauds = countFrauds(bestFraud, k);

        // Adaptive refinement: a boundary verdict (fraud-count in [REFINE_MIN,REFINE_MAX])
        // is one neighbor from flipping, so extend the SAME heap over the next clusters
        // out to maxp. Continuing the heap yields exactly a uniform maxp scan, but only
        // boundary queries (~3% of traffic) pay for it.
        if (maxp > basep && frauds >= REFINE_MIN_FRAUDS && frauds <= REFINE_MAX_FRAUDS) {
            scanClusters(query, s, cid, basep, maxp, k);
            frauds = countFrauds(bestFraud, k);
        }
        return frauds;
    }

    /**
     * Scans probed clusters {@code cid[from..to)} into the caller's top-k heap,
     * carrying the running {@code worstNeighbor}/{@code scanned} cursor through
     * {@code s} so a later pass can extend the same neighborhood. Honors
     * {@code scanCap} across both passes.
     */
    private void scanClusters(short[] query, Scratch s, int[] cid, int from, int to, int k) {
        short[] buf = s.buf;
        long[] bestDist = s.bestDist;
        boolean[] bestFraud = s.bestFraud;
        int worstNeighbor = s.worstNeighbor;
        int scanned = s.scanned;
        // DIMS is fixed at 14 in this challenge; keep query lanes in locals so the
        // inner loop only loads candidates from buf.
        int q0 = query[0];
        int q1 = query[1];
        int q2 = query[2];
        int q3 = query[3];
        int q4 = query[4];
        int q5 = query[5];
        int q6 = query[6];
        int q7 = query[7];
        int q8 = query[8];
        int q9 = query[9];
        int q10 = query[10];
        int q11 = query[11];
        int q12 = query[12];
        int q13 = query[13];
        scan:
        for (int j = from; j < to; j++) {
            int c = cid[j];
            int start = dataset.clusterStart(c);
            int end = dataset.clusterEnd(c);
            for (int base = start; base < end; base += CHUNK) {
                int remaining = scanCap - scanned;
                if (remaining <= 0) {
                    break scan;
                }
                int n = Math.min(Math.min(CHUNK, end - base), remaining);
                dataset.copyVectorRange(base, n, buf, 0);
                for (int t = 0; t < n; t++) {
                    int off = t * DIMS;
                    int a0 = q0 - buf[off];
                    int a1 = q1 - buf[off + 1];
                    int a2 = q2 - buf[off + 2];
                    int a3 = q3 - buf[off + 3];
                    int a4 = q4 - buf[off + 4];
                    int a5 = q5 - buf[off + 5];
                    int a6 = q6 - buf[off + 6];
                    int a7 = q7 - buf[off + 7];
                    int a8 = q8 - buf[off + 8];
                    int a9 = q9 - buf[off + 9];
                    int a10 = q10 - buf[off + 10];
                    int a11 = q11 - buf[off + 11];
                    int a12 = q12 - buf[off + 12];
                    int a13 = q13 - buf[off + 13];

                    long sum0 = (long) a0 * a0 + (long) a1 * a1 + (long) a2 * a2 + (long) a3 * a3;
                    long sum1 = (long) a4 * a4 + (long) a5 * a5 + (long) a6 * a6 + (long) a7 * a7;
                    long sum2 = (long) a8 * a8 + (long) a9 * a9 + (long) a10 * a10 + (long) a11 * a11;
                    long sum3 = (long) a12 * a12 + (long) a13 * a13;
                    long d2 = sum0 + sum1 + sum2 + sum3;
                    if (d2 < bestDist[worstNeighbor]) {
                        bestDist[worstNeighbor] = d2;
                        bestFraud[worstNeighbor] = dataset.isFraud(base + t);
                        worstNeighbor = indexOfMaxTopK(bestDist, k);
                    }
                }
                scanned += n;
            }
        }
        s.worstNeighbor = worstNeighbor;
        s.scanned = scanned;
    }

    private static int countFrauds(boolean[] bestFraud, int k) {
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

    /**
     * Per-request working memory, recycled through {@link #scratchPool}. Sizing
     * the pool to the permit count means a request always finds a free Scratch
     * once it is admitted, so the steady-state scoring path is allocation-free.
     */
    private static final class Scratch {
        final double[] qvec = new double[DIMS];
        final short[] query = new short[DIMS];
        final long[] cdist;
        final int[] cid;
        final long[] bestDist = new long[K];
        final boolean[] bestFraud = new boolean[K];
        final short[] buf = new short[CHUNK * DIMS];
        // Running cursor carried between the baseline and refinement scan passes.
        int worstNeighbor;
        int scanned;

        Scratch(int maxNprobe) {
            this.cdist = new long[maxNprobe];
            this.cid = new int[maxNprobe];
        }
    }

    /** Insertion sort of the first {@code len} (distance, clusterId) pairs by ascending distance. */
    private static void sortByDistance(long[] dist, int[] id, int len) {
        for (int i = 1; i < len; i++) {
            long d = dist[i];
            int c = id[i];
            int j = i - 1;
            while (j >= 0 && dist[j] > d) {
                dist[j + 1] = dist[j];
                id[j + 1] = id[j];
                j--;
            }
            dist[j + 1] = d;
            id[j + 1] = c;
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

    /**
     * Top-k heap here is effectively fixed at 5. Keep a tiny specialized fast
     * path for the per-candidate hot loop, with fallback for tests/smaller sets.
     */
    private static int indexOfMaxTopK(long[] values, int len) {
        if (len == 5) {
            int max = values[1] > values[0] ? 1 : 0;
            if (values[2] > values[max]) {
                max = 2;
            }
            if (values[3] > values[max]) {
                max = 3;
            }
            if (values[4] > values[max]) {
                max = 4;
            }
            return max;
        }
        return indexOfMax(values, len);
    }

    private static int resolveInt(String envVar, int defaultValue) {
        String env = System.getenv(envVar);
        if (env != null && !env.isBlank()) {
            try {
                return Integer.parseInt(env.trim());
            } catch (NumberFormatException e) {
                LOG.warning(() -> "Invalid " + envVar + " '" + env + "'; using default " + defaultValue);
            }
        }
        return defaultValue;
    }
}
