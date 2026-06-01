package io.github.brunoborges.rinha2026;

import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Semaphore;
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

    /** Number of clusters probed per query unless overridden by {@code NPROBE}. */
    public static final int DEFAULT_NPROBE = 24;

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
    private final Semaphore scanPermits;
    private final ArrayBlockingQueue<short[]> bufferPool;
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
        int permits = Math.max(1, Runtime.getRuntime().availableProcessors());
        this.scanPermits = new Semaphore(permits);
        this.bufferPool = new ArrayBlockingQueue<>(permits);
        for (int i = 0; i < permits; i++) {
            this.bufferPool.add(new short[CHUNK * DIMS]);
        }
        LOG.info(() -> "IVF scorer: clusters=" + dataset.clusters() + ", nprobe=" + this.nprobe
                + ", scanCap=" + this.scanCap + ", scanThreads=" + permits);
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
     * Package-private hook used by tests to exercise the IVF candidate search
     * directly with a quantized query, bypassing vectorization.
     */
    int countFraudsAmongNearest(short[] query) {
        int clusters = dataset.clusters();
        int p = Math.min(nprobe, clusters);

        // Find the p nearest centroids (bounded buffer of (distance, clusterId)).
        long[] cdist = new long[p];
        int[] cid = new int[p];
        Arrays.fill(cdist, Long.MAX_VALUE);
        int worst = 0;
        for (int c = 0; c < clusters; c++) {
            long d = dataset.centroidSquaredDistance(query, c);
            if (d < cdist[worst]) {
                cdist[worst] = d;
                cid[worst] = c;
                worst = indexOfMax(cdist);
            }
        }
        sortByDistance(cdist, cid);

        int count = dataset.count();
        int k = Math.min(K, count);
        long[] bestDist = new long[k];
        boolean[] bestFraud = new boolean[k];
        Arrays.fill(bestDist, Long.MAX_VALUE);
        int worstNeighbor = 0;

        scanPermits.acquireUninterruptibly();
        short[] buf = null;
        try {
            buf = bufferPool.take();
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
                            worstNeighbor = indexOfMax(bestDist);
                        }
                    }
                    scanned += n;
                }
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        } finally {
            if (buf != null) {
                bufferPool.offer(buf);
            }
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

    /** Insertion sort of the (distance, clusterId) pairs by ascending distance. */
    private static void sortByDistance(long[] dist, int[] id) {
        for (int i = 1; i < dist.length; i++) {
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

    private static int indexOfMax(long[] values) {
        int max = 0;
        for (int i = 1; i < values.length; i++) {
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
