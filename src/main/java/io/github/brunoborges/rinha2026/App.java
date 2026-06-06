package io.github.brunoborges.rinha2026;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

import io.github.brunoborges.rinha2026.kdport.KdTree;
import io.github.brunoborges.rinha2026.kdport.KdTreeIO;

/**
 * Fraud-scoring API for Rinha de Backend 2026.
 *
 * <p>This process does <em>not</em> bind a public TCP port. The Rust load balancer (lapada-style) owns
 * {@code :9999}, accepts every client connection, and hands the accepted socket file descriptor to this
 * process over a Unix control socket via {@code SCM_RIGHTS}. {@link FdEpollServer} then serves the
 * request directly on that fd — the balancer is never on the data path.
 *
 * <ul>
 *   <li>{@code GET /ready} &mdash; readiness probe, returns 200 once the dataset is faulted in.</li>
 *   <li>{@code POST /fraud-score} &mdash; scores a transaction for fraud.</li>
 * </ul>
 */
public class App {

    private static final Logger LOG = Logger.getLogger(App.class.getName());

    /** Default location of the pre-built binary reference dataset. */
    private static final String DEFAULT_REFERENCES_BIN = "resources/references.bin";

    /** Default location of the pre-built mmap-loadable KdTree index binary. */
    private static final String DEFAULT_KDTREE_BIN = "resources/kdtree.bin";

    /** Default location of the JSON reference dataset (fallback). */
    private static final String DEFAULT_REFERENCES_FILE = "resources/references.json.gz";

    /** Default Unix control socket this instance binds to receive client fds from the load balancer. */
    private static final String DEFAULT_FD_SOCKET = "/sockets/api.sock";

    /** Default readiness marker file the container healthcheck waits on. */
    private static final String DEFAULT_READY_FILE = "/tmp/rinha-ready";

    private final FraudScorer scorer;

    public App() {
        this(loadDefaultScorer());
    }

    public App(FraudScorer scorer) {
        this.scorer = scorer;
    }

    public static void main(String[] args) throws Exception {
        // Offline benchmark dispatch: BENCH=1 (or `bench` as the first arg) runs the HTTP-free
        // vector-search benchmark in this same image instead of starting the server.
        String bench = System.getenv("BENCH");
        if ((bench != null && bench.trim().equals("1"))
                || (args.length > 0 && "bench".equals(args[0]))) {
            BenchmarkCli.run(args);
            return;
        }
        // AOT cache training (-XX:AOTMode=record): drive the scoring hot path with synthetic
        // requests, then exit, so the JVM records production classes + method profiles.
        String train = System.getenv("AOT_TRAINING");
        if ((train != null && train.trim().equals("1"))
                || (args.length > 0 && "train".equals(args[0]))) {
            TrainingDriver.run();
            return;
        }
        new App().serve();
        // The epoll loop runs on a non-daemon thread; main can return while it keeps serving.
    }

    /**
     * Starts the fd-passing server: faults the dataset into the page cache, binds the Unix control
     * socket, then publishes the readiness marker so the load balancer and healthcheck proceed.
     */
    public void serve() throws Exception {
        preload();

        HttpRouter router = new HttpRouter(scorer);

        String socketPath = env("FD_SOCKET", DEFAULT_FD_SOCKET);
        FdEpollServer server = new FdEpollServer(socketPath, router);
        server.start(); // blocks until the control socket is bound and listening

        warmup(router);

        router.markReady();
        writeReadyFile();
        LOG.info(() -> "Serving fraud-score API; control socket " + socketPath);
    }

    /**
     * Drives synthetic requests through the hot path to provoke background C2
     * compilation, then publishes readiness only once the compiler has gone quiet
     * — so the single contest cold run hits C2-compiled code instead of paying a
     * fat interpreter/C1 p99 tail. Tunable at runtime (no rebuild) via env:
     * {@code WARMUP_MS} (budget cap, default 30000; 0 disables), {@code
     * WARMUP_STABLE_MS} (compiler-quiet window, default 3000), {@code
     * WARMUP_MAX_ITERS} (safety cap, default 5_000_000).
     */
    private void warmup(HttpRouter router) {
        long budgetMs = envInt("WARMUP_MS", 30_000);
        if (budgetMs <= 0) {
            LOG.info("JIT warmup disabled (WARMUP_MS=0)");
            return;
        }
        long stableMs = envInt("WARMUP_STABLE_MS", 3_000);
        int maxIters = envInt("WARMUP_MAX_ITERS", 5_000_000);
        try {
            TrainingDriver.warmup(router, budgetMs, stableMs, maxIters);
        } catch (Exception e) {
            LOG.warning(() -> "JIT warmup failed (continuing anyway): " + e.getMessage());
        }
    }

    private static int envInt(String name, int dflt) {
        String v = System.getenv(name);
        if (v == null) {
            return dflt;
        }
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            return dflt;
        }
    }

    /**
     * Builds the production scorer backed by the reference dataset. Prefers a pre-built, memory-mapped
     * {@code .bin} (off-heap, shared across instances); falls back to streaming the JSON dataset, and
     * finally to {@link StubFraudScorer} (with a loud warning) so the server still starts in development
     * and tests.
     */
    static FraudScorer loadDefaultScorer() {
        // Diagnostic override for bottleneck attribution (see VectorizeOnlyFraudScorer): SCORER=stub
        // isolates the HTTP/parse/serialize path, SCORER=vectorize adds feature extraction, default/ivf
        // is the full search. Only honoured when explicitly set.
        String mode = System.getenv("SCORER");
        if (mode != null) {
            mode = mode.trim().toLowerCase();
            if (mode.equals("stub")) {
                LOG.warning("SCORER=stub: serving constant responses (HTTP-path-only diagnostic).");
                return new StubFraudScorer();
            }
            if (mode.equals("vectorize")) {
                LOG.warning("SCORER=vectorize: vectorizing only, skipping the scan (diagnostic).");
                return new VectorizeOnlyFraudScorer(new TransactionVectorizer());
            }
        }

        // KdTree exact-kNN index is the default when its binary is present; SCORER=ivf forces the
        // approximate IVF scan (fallback). Offline eval: KdTree E=0 vs IVF E=26 at ~10x fewer visits.
        boolean forceIvf = "ivf".equals(mode);
        boolean wantKd = "kdtree".equals(mode) || "kd".equals(mode);
        if (!forceIvf) {
            Path kdBin = resolvePath("KDTREE_BIN", DEFAULT_KDTREE_BIN);
            if (Files.isReadable(kdBin)) {
                try {
                    long start = System.nanoTime();
                    KdTree tree = KdTreeIO.loadMmap(kdBin);
                    long ms = (System.nanoTime() - start) / 1_000_000;
                    LOG.info(() -> "Memory-mapped KdTree index (" + tree.size() + " nodes) from '"
                            + kdBin + "' in " + ms + " ms");
                    return new KdTreeFraudScorer(new TransactionVectorizer(), tree);
                } catch (IOException e) {
                    LOG.warning(() -> "Failed to load KdTree index '" + kdBin + "': " + e.getMessage()
                            + "; trying IVF/dataset.");
                }
            } else if (wantKd) {
                LOG.warning(() -> "SCORER=kdtree requested but no readable index at '"
                        + kdBin.toAbsolutePath() + "'; trying IVF/dataset.");
            }
        }

        Path bin = resolvePath("REFERENCES_BIN", DEFAULT_REFERENCES_BIN);
        if (Files.isReadable(bin)) {
            try {
                long start = System.nanoTime();
                ReferenceDataset dataset = ReferenceDataset.mmap(bin);
                long ms = (System.nanoTime() - start) / 1_000_000;
                LOG.info(() -> "Memory-mapped " + dataset.count() + " reference vectors from '"
                        + bin + "' in " + ms + " ms"
                        + (dataset.hasIndex() ? " (IVF index: " + dataset.clusters() + " clusters)" : ""));
                TransactionVectorizer vectorizer = new TransactionVectorizer();
                return dataset.hasIndex()
                        ? new IvfFraudScorer(vectorizer, dataset)
                        : new VectorSearchFraudScorer(vectorizer, dataset);
            } catch (IOException e) {
                LOG.warning(() -> "Failed to memory-map '" + bin + "': " + e.getMessage()
                        + "; trying JSON dataset.");
            }
        }

        Path json = resolvePath("REFERENCES_FILE", DEFAULT_REFERENCES_FILE);
        if (Files.isReadable(json)) {
            try {
                long start = System.nanoTime();
                ReferenceDataset dataset = ReferenceDataset.loadFromFile(json);
                long ms = (System.nanoTime() - start) / 1_000_000;
                LOG.info(() -> "Loaded " + dataset.count() + " reference vectors from '"
                        + json + "' in " + ms + " ms (consider pre-building a .bin via ReferenceConverter)");
                return new VectorSearchFraudScorer(new TransactionVectorizer(), dataset);
            } catch (IOException e) {
                LOG.warning(() -> "Failed to load reference dataset from '" + json
                        + "': " + e.getMessage() + "; falling back to StubFraudScorer.");
            }
        }

        LOG.warning(() -> "No reference dataset found (looked for '" + bin.toAbsolutePath()
                + "' and '" + json.toAbsolutePath() + "'); falling back to StubFraudScorer."
                + " Set REFERENCES_BIN or REFERENCES_FILE to enable real fraud scoring."
                + " THIS WILL RETURN PLACEHOLDER SCORES.");
        return new StubFraudScorer();
    }

    /**
     * Page-faults the entire memory-mapped reference dataset into the OS page cache before readiness is
     * published. The eval starts containers cold; without this the first live requests pay major page
     * faults synchronously on a single CPU, backing up the queue into an error storm.
     *
     * <p>Page-faulting the dataset is paired with {@link #warmup} (JIT warmup);
     * together they make the single cold contest run hit resident pages and
     * C2-compiled code. Failures are logged and swallowed so a hiccup never
     * prevents the server from starting.
     */
    private void preload() {
        try {
            long start = System.nanoTime();
            scorer.preload();
            long ms = (System.nanoTime() - start) / 1_000_000;
            LOG.info(() -> "Pre-faulted reference dataset into page cache in " + ms + " ms");
        } catch (Exception e) {
            LOG.warning(() -> "Scorer preload failed (continuing anyway): " + e.getMessage());
        }
    }

    private void writeReadyFile() {
        String path = env("READY_FILE", DEFAULT_READY_FILE);
        try {
            Files.writeString(Path.of(path), "ready");
        } catch (IOException e) {
            LOG.warning(() -> "Failed to write ready file '" + path + "': " + e.getMessage());
        }
    }

    private static Path resolvePath(String envVar, String defaultPath) {
        String configured = System.getenv(envVar);
        return Path.of(configured != null && !configured.isBlank() ? configured.trim() : defaultPath);
    }

    private static String env(String name, String defaultValue) {
        String v = System.getenv(name);
        return (v != null && !v.isBlank()) ? v.trim() : defaultValue;
    }
}
