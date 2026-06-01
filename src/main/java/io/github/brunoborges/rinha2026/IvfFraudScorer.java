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
     * Number of clusters probed per query unless overridden by {@code NPROBE}.
     *
     * <p>Tuned empirically against the official load test (see docs/EVALUATION.md
     * scoring). Detection quality is effectively cap-saturated by ~8 probes
     * (rate_component stays pinned at its 3000 ceiling), so probing more clusters
     * buys only a tiny absolute-penalty reduction while scan cost — and therefore
     * p99 latency — grows linearly. Past the CPU-saturation knee, the tail latency
     * blows up super-linearly. NPROBE=8 sits just below that knee: in the native
     * sweep it scored ~3970 vs ~3060 at NPROBE=24 (a ~900-point gain almost
     * entirely from p99), with failure_rate ~0.0005 (far under the 15% cutoff).
     */
    public static final int DEFAULT_NPROBE = 8;

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
    private final int nprobe;
    private final int scanCap;

    public IvfFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset) {
        this(vectorizer, dataset, resolveInt("NPROBE", DEFAULT_NPROBE),
                resolveInt("SCAN_CAP", DEFAULT_SCAN_CAP));
    }

    public IvfFraudScorer(TransactionVectorizer vectorizer, ReferenceDataset dataset,
                          int nprobe, int scanCap) {
        if (!dataset.hasIndex()) {
            throw new IllegalArgumentException("dataset has no IVF index; use VectorSearchFraudScorer");
        }
        this.vectorizer = vectorizer;
        this.dataset = dataset;
        this.nprobe = Math.max(1, Math.min(nprobe, dataset.clusters()));
        this.scanCap = Math.max(K, scanCap);
        // One reusable Scratch per permit; the bounded pool both caps concurrent
        // scans (take() blocks when empty) and recycles every per-request buffer,
        // so scoring allocates nothing on the heap after warmup. Sized by WORKERS
        // (not availableProcessors, which the cgroup pins to 1) so several
        // memory-stall-bound scans can overlap within the CPU quota.
        int permits = Concurrency.workers();
        this.scratchPool = new ArrayBlockingQueue<>(permits);
        for (int i = 0; i < permits; i++) {
            this.scratchPool.add(new Scratch(this.nprobe));
        }
        LOG.info(() -> "IVF scorer: clusters=" + dataset.clusters() + ", nprobe=" + this.nprobe
                + ", scanCap=" + this.scanCap + ", scanThreads=" + permits);
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
        int p = Math.min(nprobe, clusters);

        // Find the p nearest centroids (bounded buffer of (distance, clusterId)).
        long[] cdist = s.cdist;
        int[] cid = s.cid;
        Arrays.fill(cdist, 0, p, Long.MAX_VALUE);
        int worst = 0;
        for (int c = 0; c < clusters; c++) {
            long d = dataset.centroidSquaredDistance(query, c);
            if (d < cdist[worst]) {
                cdist[worst] = d;
                cid[worst] = c;
                worst = indexOfMax(cdist, p);
            }
        }
        sortByDistance(cdist, cid, p);

        int count = dataset.count();
        int k = Math.min(K, count);
        long[] bestDist = s.bestDist;
        boolean[] bestFraud = s.bestFraud;
        Arrays.fill(bestDist, 0, k, Long.MAX_VALUE);
        Arrays.fill(bestFraud, 0, k, false);
        int worstNeighbor = 0;

        short[] buf = s.buf;
        int scanned = 0;
        scan:
        for (int j = 0; j < p; j++) {
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
                    long d2 = squaredDistance(query, buf, t * DIMS);
                    if (d2 < bestDist[worstNeighbor]) {
                        bestDist[worstNeighbor] = d2;
                        bestFraud[worstNeighbor] = dataset.isFraud(base + t);
                        worstNeighbor = indexOfMax(bestDist, k);
                    }
                }
                scanned += n;
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

        Scratch(int nprobe) {
            this.cdist = new long[nprobe];
            this.cid = new int[nprobe];
        }
    }

    /**
     * Squared Euclidean distance (quantized units) between {@code query} and the
     * vector packed at {@code off} in a heap buffer. Mirrors
     * {@link ReferenceDataset#squaredDistance} but reads the candidate from a
     * plain {@code short[]} (intrinsified, auto-vectorizable) rather than via
     * per-element off-heap FFM access. Two accumulators break the dependency
     * chain so the multiply-adds pipeline.
     */
    private static long squaredDistance(short[] query, short[] buf, int off) {
        long sum0 = 0;
        long sum1 = 0;
        for (int d = 0; d < DIMS; d += 2) {
            int a = query[d] - buf[off + d];
            int b = query[d + 1] - buf[off + d + 1];
            sum0 += (long) a * a;
            sum1 += (long) b * b;
        }
        return sum0 + sum1;
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
