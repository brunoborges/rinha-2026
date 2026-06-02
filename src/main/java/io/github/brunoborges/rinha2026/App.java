package io.github.brunoborges.rinha2026;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.Executors;
import java.util.logging.Logger;

import static java.nio.charset.StandardCharsets.UTF_8;

/**
 * Embedded HTTP server exposing the fraud-scoring API for Rinha de Backend 2026.
 *
 * <ul>
 *   <li>{@code GET /ready} &mdash; readiness probe, returns 200 when serving.</li>
 *   <li>{@code POST /fraud-score} &mdash; scores a transaction for fraud.</li>
 * </ul>
 */
public class App {

    private static final Logger LOG = Logger.getLogger(App.class.getName());

    private static final int DEFAULT_PORT = 9999;

    /** Default location of the pre-built binary reference dataset. */
    private static final String DEFAULT_REFERENCES_BIN = "resources/references.bin";

    /** Default location of the JSON reference dataset (fallback). */
    private static final String DEFAULT_REFERENCES_FILE = "resources/references.json.gz";

    private static final byte[] READY_BODY = "{\"status\":\"ready\"}".getBytes(UTF_8);

    private final FraudScorer scorer;
    private final RequestVectorParser parser;

    public App() {
        this(loadDefaultScorer());
    }

    public App(FraudScorer scorer) {
        this.scorer = scorer;
        this.parser = scorer.supportsVectorInput()
                ? new RequestVectorParser(scorer.vectorizer())
                : null;
    }

    /**
     * Builds the production scorer backed by the reference dataset. Prefers a
     * pre-built, memory-mapped {@code .bin} (off-heap, shared across instances);
     * falls back to streaming the JSON dataset, and finally to
     * {@link StubFraudScorer} (with a loud warning) so the server still starts in
     * development and tests.
     */
    private static FraudScorer loadDefaultScorer() {
        // Diagnostic override for bottleneck attribution (see
        // VectorizeOnlyFraudScorer): SCORER=stub isolates the HTTP/parse/serialize
        // path, SCORER=vectorize adds feature extraction, default/ivf is the full
        // search. Only honoured when explicitly set.
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

    private static Path resolvePath(String envVar, String defaultPath) {
        String configured = System.getenv(envVar);
        return Path.of(configured != null && !configured.isBlank() ? configured.trim() : defaultPath);
    }

    public static void main(String[] args) throws IOException {
        // Offline benchmark dispatch: BENCH=1 (or `bench` as the first arg) runs the
        // HTTP-free vector-search benchmark in this same (native) image instead of
        // starting the server. Production never sets BENCH, so the server path is the
        // default. Kept reachable from main() so native-image includes BenchmarkCli.
        String bench = System.getenv("BENCH");
        if ((bench != null && bench.trim().equals("1"))
                || (args.length > 0 && "bench".equals(args[0]))) {
            BenchmarkCli.run(args);
            return;
        }
        HttpServer server = new App().start(resolvePort(args));
        System.out.println("Listening on http://localhost:" + server.getAddress().getPort());
    }

    /**
     * Creates and starts the HTTP server.
     *
     * @param port the TCP port to bind, or {@code 0} for an ephemeral port
     * @return the running {@link HttpServer}
     */
    public HttpServer start(int port) throws IOException {
        preload();
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/ready", this::handleReady);
        server.createContext("/fraud-score", this::handleFraudScore);
        server.setExecutor(Executors.newVirtualThreadPerTaskExecutor());
        server.start();
        return server;
    }

    /**
     * Page-faults the entire memory-mapped reference dataset into the OS page
     * cache before the port opens. The eval starts containers cold; without this
     * the first live requests pay major page faults synchronously on a single CPU,
     * backing up the queue into an error storm.
     *
     * <p>This is the only pre-serving work needed: compiled ahead-of-time as a
     * GraalVM Native Image there is no JIT to warm, so the previous scan-loop
     * warmup is gone. Failures are logged and swallowed so a hiccup never prevents
     * the server from starting.
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

    private void handleReady(HttpExchange exchange) throws IOException {
        try {
            if (!"GET".equals(exchange.getRequestMethod())) {
                sendError(exchange, 405, "method not allowed");
                return;
            }
            send(exchange, 200, READY_BODY);
        } finally {
            exchange.close();
        }
    }

    private void handleFraudScore(HttpExchange exchange) throws IOException {
        try {
            if (!"POST".equals(exchange.getRequestMethod())) {
                sendError(exchange, 405, "method not allowed");
                return;
            }
            if (parser != null) {
                scoreStreaming(exchange);
            } else {
                scoreFromRecord(exchange);
            }
        } finally {
            exchange.close();
        }
    }

    /**
     * Hot path: parse the body straight into a pooled feature vector and score it,
     * allocating no {@link FraudRequest} graph. Any malformed input (bad JSON,
     * missing required section, invalid timestamp) surfaces as an
     * {@link IOException} and is reported as {@code HTTP 400}, matching the
     * record path's rejection behaviour.
     */
    private void scoreStreaming(HttpExchange exchange) throws IOException {
        RequestVectorParser.State st = parser.acquire();
        FraudResponse response;
        try {
            parser.vectorize(exchange.getRequestBody(), st);
            response = scorer.scoreVector(st.qvec);
        } catch (IOException e) {
            sendError(exchange, 400, e.getMessage());
            return;
        } finally {
            // qvec is consumed by scoreVector, so the State can be recycled before
            // the (potentially slow) response write — keeping the parser pool free
            // to admit the next request rather than gating on the network.
            parser.release(st);
        }
        send(exchange, 200, scoreBody(response));
    }

    /**
     * Fallback path for scorers that do not support vector input (the {@code stub}
     * and {@code vectorize} diagnostic modes): parse into a {@link FraudRequest},
     * validate, and score.
     */
    private void scoreFromRecord(HttpExchange exchange) throws IOException {
        FraudRequest request;
        try {
            request = FraudRequestParser.parse(exchange.getRequestBody());
        } catch (IOException e) {
            sendError(exchange, 400, "invalid JSON: " + e.getMessage());
            return;
        }

        String validationError = validate(request);
        if (validationError != null) {
            sendError(exchange, 400, validationError);
            return;
        }

        FraudResponse response = scorer.score(request);
        send(exchange, 200, scoreBody(response));
    }

    /**
     * Validates that the mandatory sections of the payload are present.
     * {@code last_transaction} is intentionally allowed to be {@code null}.
     *
     * @return an error message, or {@code null} when the request is valid
     */
    private static String validate(FraudRequest request) {
        if (request == null) {
            return "request body must be a JSON object";
        }
        if (request.id() == null) {
            return "missing required field: id";
        }
        if (request.transaction() == null) {
            return "missing required field: transaction";
        }
        if (request.customer() == null) {
            return "missing required field: customer";
        }
        if (request.merchant() == null) {
            return "missing required field: merchant";
        }
        if (request.terminal() == null) {
            return "missing required field: terminal";
        }
        return null;
    }

    private static int resolvePort(String[] args) {
        if (args.length > 0) {
            return Integer.parseInt(args[0]);
        }
        String env = System.getenv("PORT");
        return (env != null && !env.isBlank()) ? Integer.parseInt(env.trim()) : DEFAULT_PORT;
    }

    private void sendError(HttpExchange exchange, int status, String message) throws IOException {
        String json = "{\"error\":\"" + escape(message) + "\"}";
        send(exchange, status, json.getBytes(UTF_8));
    }

    private static byte[] scoreBody(FraudResponse response) {
        String json = "{\"approved\":" + response.approved()
                + ",\"fraud_score\":" + response.fraudScore() + "}";
        return json.getBytes(UTF_8);
    }

    private void send(HttpExchange exchange, int status, byte[] body) throws IOException {
        exchange.getResponseHeaders().set("Content-Type", "application/json; charset=utf-8");
        exchange.sendResponseHeaders(status, body.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(body);
        }
    }

    /** Minimal JSON string escaping for the small, hand-written error responses. */
    private static String escape(String s) {
        if (s == null) {
            return "";
        }
        StringBuilder b = new StringBuilder(s.length() + 8);
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                default -> {
                    if (c < 0x20) {
                        b.append(String.format("\\u%04x", (int) c));
                    } else {
                        b.append(c);
                    }
                }
            }
        }
        return b.toString();
    }
}
