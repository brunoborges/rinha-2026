package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Offline, HTTP-free benchmark for the vector-search hot path.
 *
 * <p>Its sole purpose is to make the IVF scan ({@link IvfFraudScorer}) as fast as
 * possible in isolation: no {@code com.sun.net.httpserver}, no sockets, no
 * load generator. It memory-maps the same {@code references.bin} the production
 * server uses, replays the labelled requests from {@code test/test-data.json}
 * through the production parse/score code, and reports throughput, a latency
 * distribution, and the detection confusion matrix so a scan optimization can be
 * judged on both speed <em>and</em> accuracy.
 *
 * <p>It is invoked from {@link App#main(String[])} when {@code BENCH=1} (or the
 * first CLI arg is {@code bench}) so it runs inside the very same GraalVM Native
 * Image as production &mdash; important because the AOT-compiled scan has
 * different performance characteristics from a JIT-warmed JVM (notably FFM
 * memory-segment access is not intrinsified under native-image), so a plain JVM
 * harness would mislead.
 *
 * <h2>Modes ({@code BENCH_MODE})</h2>
 * <ul>
 *   <li>{@code scan} (default) &mdash; pre-quantizes every request and times only
 *       {@link IvfFraudScorer#countFraudsAmongNearest(short[])}, the allocation-free
 *       IVF candidate search. This is the purest measure of "vector search" speed.</li>
 *   <li>{@code score} &mdash; pre-vectorizes every request and times
 *       {@link IvfFraudScorer#scoreVector(double[])} (quantize + scan + response
 *       allocation), the per-request work the HTTP handler does after parsing.</li>
 *   <li>{@code full} &mdash; retains the request JSON and times parse + vectorize +
 *       score, i.e. everything except the network.</li>
 * </ul>
 *
 * <h2>Knobs (environment variables)</h2>
 * <ul>
 *   <li>{@code TEST_DATA} &mdash; path to the labelled corpus (default {@code test/test-data.json}).</li>
 *   <li>{@code REFERENCES_BIN} &mdash; path to the mmap dataset (default {@code resources/references.bin}).</li>
 *   <li>{@code BENCH_MODE} &mdash; {@code scan} | {@code score} | {@code full}.</li>
 *   <li>{@code BENCH_PASSES} &mdash; timed passes over the corpus (default 5).</li>
 *   <li>{@code BENCH_LIMIT} &mdash; cap entries loaded (0 = all); handy for quick host runs.</li>
 *   <li>{@code THREADS} &mdash; concurrent scan workers (default {@code WORKERS}).</li>
 *   <li>{@code EXECUTOR} &mdash; {@code platform} (default) or {@code virtual}. Swaps the
 *       workers between platform and virtual threads. The IVF scan never yields, so under
 *       the 1-CPU cgroup virtual workers share a single carrier and tend to serialize;
 *       platform threads let the OS overlap memory stalls. Provided so the trade-off can
 *       be measured rather than assumed.</li>
 *   <li>{@code THROTTLE_STATS} &mdash; {@code auto} (default) | {@code off}. When the process
 *       runs inside a CPU-bandwidth-limited cgroup (e.g. {@code docker --cpus=1.0}), reports the
 *       Linux CFS throttle counters ({@code nr_periods} / {@code nr_throttled} / throttled time)
 *       measured across the timed passes. This shows how often a given {@code THREADS}/{@code WORKERS}
 *       setting burns the whole quota inside a 100ms period and gets frozen for the remainder &mdash;
 *       i.e. how hard it is hitting the quota wall. {@code auto} = print when the cgroup
 *       {@code cpu.stat} is readable; {@code off} = never read it.</li>
 *   <li>{@code NPROBE} / {@code MAX_NPROBE} / {@code SCAN_CAP} / {@code WORKERS} &mdash; passed
 *       straight through to {@link IvfFraudScorer} / {@link Concurrency}.</li>
 * </ul>
 */
final class BenchmarkCli {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private static final String DEFAULT_BIN = "resources/references.bin";
    private static final String DEFAULT_TEST_DATA = "test/test-data.json";

    /** Op-indices claimed per atomic fetch, to keep cursor contention off the hot path. */
    private static final int CLAIM = 256;

    /** Blackhole sink: prevents the AOT compiler from eliminating the scan as dead code. */
    private static volatile long blackhole;

    private BenchmarkCli() {
    }

    static void run(String[] args) {
        try {
            new BenchmarkCli().execute();
        } catch (Exception e) {
            System.err.println("benchmark failed: " + e);
            e.printStackTrace();
            System.exit(1);
        }
    }

    private enum Mode { SCAN, SCORE, FULL }

    private void execute() throws IOException {
        Mode mode = parseMode(env("BENCH_MODE", "scan"));
        Path bin = Path.of(env("REFERENCES_BIN", DEFAULT_BIN));
        Path data = Path.of(env("TEST_DATA", DEFAULT_TEST_DATA));
        int passes = Math.max(1, envInt("BENCH_PASSES", 5));
        int limit = Math.max(0, envInt("BENCH_LIMIT", 0));
        int workers = Concurrency.workers();
        int threads = Math.max(1, envInt("THREADS", workers));
        boolean useVirtual = "virtual".equalsIgnoreCase(env("EXECUTOR", "platform"));
        boolean throttleStats = !"off".equalsIgnoreCase(env("THROTTLE_STATS", "auto"));

        if (!Files.isReadable(bin)) {
            throw new IOException("reference dataset not readable: " + bin.toAbsolutePath()
                    + " (set REFERENCES_BIN)");
        }
        if (!Files.isReadable(data)) {
            throw new IOException("test data not readable: " + data.toAbsolutePath()
                    + " (set TEST_DATA)");
        }

        long t0 = System.nanoTime();
        ReferenceDataset dataset = ReferenceDataset.mmap(bin);
        if (!dataset.hasIndex()) {
            throw new IOException("dataset has no IVF index; rebuild references.bin with ReferenceConverter");
        }
        TransactionVectorizer vectorizer = new TransactionVectorizer();
        IvfFraudScorer scorer = new IvfFraudScorer(vectorizer, dataset);
        scorer.preload();
        long loadMs = (System.nanoTime() - t0) / 1_000_000;

        Corpus corpus = loadCorpus(data, vectorizer, mode, limit);
        int n = corpus.n;
        if (n == 0) {
            throw new IOException("no entries loaded from " + data);
        }

        // Configuration banner: reproducibility matters more than terse output here.
        System.out.println("=== vector-search benchmark ===");
        System.out.println("mode        : " + mode.name().toLowerCase());
        System.out.println("dataset     : " + dataset.count() + " vectors, "
                + dataset.clusters() + " clusters (mmapped in " + loadMs + " ms)");
        System.out.println("corpus      : " + n + " requests from " + data.getFileName()
                + (limit > 0 ? " (capped at " + limit + ")" : ""));
        System.out.println("passes      : " + passes + "  (total ops: " + ((long) passes * n) + ")");
        System.out.println("threads     : " + threads + "   workers(pool)=" + workers
                + (threads > workers ? "  [WARN: THREADS>WORKERS -> latency includes pool blocking]" : ""));
        System.out.println("executor    : " + (useVirtual ? "virtual" : "platform")
                + (useVirtual ? "  [NOTE: scan never yields; vthreads share ~"
                        + Runtime.getRuntime().availableProcessors()
                        + " carrier(s) and tend to serialize on a CPU-bound scan]" : ""));
        System.out.println("nprobe      : " + envInt("NPROBE", IvfFraudScorer.DEFAULT_NPROBE)
                + "  max=" + envInt("MAX_NPROBE", IvfFraudScorer.DEFAULT_MAX_NPROBE)
                + "  scanCap=" + envInt("SCAN_CAP", IvfFraudScorer.DEFAULT_SCAN_CAP));
        CpuStat throttleProbe = throttleStats ? readCpuStat() : CpuStat.NONE;
        System.out.println("throttle    : " + (!throttleStats ? "off (THROTTLE_STATS=off)"
                : throttleProbe.valid() ? "on (" + throttleProbe.source + ")"
                : "unavailable (no cpu.stat; not in a CPU-limited cgroup)"));
        System.out.println();

        // Correctness pass (single-threaded): tally detection against expected labels.
        // Doubles as a final page-cache / branch-predictor warmup over the whole corpus.
        Confusion cm = correctnessPass(scorer, corpus, mode);

        // Warmup pass (excluded from metrics).
        runTimed(scorer, corpus, mode, threads, useVirtual, 1, null);

        // Timed passes.
        long[] latencies = new long[passes * n];
        CpuStat throttleBefore = throttleStats ? readCpuStat() : CpuStat.NONE;
        long wallStart = System.nanoTime();
        runTimed(scorer, corpus, mode, threads, useVirtual, passes, latencies);
        long wallNs = System.nanoTime() - wallStart;
        CpuStat throttleAfter = throttleStats ? readCpuStat() : CpuStat.NONE;

        report(latencies, wallNs, cm, n);
        reportThrottle(throttleBefore, throttleAfter);
        dataset.close();
    }

    // ------------------------------------------------------------------ corpus

    /** Loaded, pre-processed requests plus their expected verdicts. */
    private static final class Corpus {
        final int n;
        final boolean[] expectedApproved;
        final short[][] queries;   // SCAN mode: pre-quantized short[14] per request
        final double[][] vectors;  // SCORE mode: pre-vectorized double[14] per request
        final byte[][] requestJson; // FULL mode: raw request bytes per request

        Corpus(int n, boolean[] expectedApproved, short[][] queries,
               double[][] vectors, byte[][] requestJson) {
            this.n = n;
            this.expectedApproved = expectedApproved;
            this.queries = queries;
            this.vectors = vectors;
            this.requestJson = requestJson;
        }
    }

    /**
     * Streams {@code test-data.json} with jackson-core (no databind), capturing each
     * entry's {@code request} sub-object via {@link JsonGenerator#copyCurrentStructure}
     * and its {@code expected_approved} label. Each request is then run through the
     * production {@link RequestVectorParser} once, and only the artifacts the chosen
     * mode needs are retained (so the resident set stays well within the native-image
     * heap budget).
     */
    private Corpus loadCorpus(Path data, TransactionVectorizer vectorizer, Mode mode, int limit)
            throws IOException {
        JsonFactory factory = new JsonFactory();
        RequestVectorParser parser = new RequestVectorParser(vectorizer);

        // Grow-able accumulators; trimmed to exact size at the end.
        int cap = 1024;
        boolean[] expected = new boolean[cap];
        short[][] queries = (mode == Mode.SCAN) ? new short[cap][] : null;
        double[][] vectors = (mode == Mode.SCORE) ? new double[cap][] : null;
        byte[][] json = (mode == Mode.FULL) ? new byte[cap][] : null;
        int n = 0;

        try (JsonParser p = factory.createParser(Files.newInputStream(data))) {
            if (p.nextToken() != JsonToken.START_OBJECT) {
                throw new IOException("test data root must be a JSON object");
            }
            while (p.nextToken() != JsonToken.END_OBJECT) {
                String top = p.currentName();
                if (!"entries".equals(top)) {
                    p.nextToken();
                    p.skipChildren();
                    continue;
                }
                if (p.nextToken() != JsonToken.START_ARRAY) {
                    throw new IOException("'entries' must be an array");
                }
                while (p.nextToken() != JsonToken.END_ARRAY) {
                    // Positioned at the START_OBJECT of one entry.
                    byte[] reqBytes = null;
                    Boolean approved = null;
                    while (p.nextToken() != JsonToken.END_OBJECT) {
                        String field = p.currentName();
                        switch (field) {
                            case "request" -> {
                                if (p.nextToken() != JsonToken.START_OBJECT) {
                                    throw new IOException("'request' must be an object");
                                }
                                ByteArrayOutputStream buf = new ByteArrayOutputStream(512);
                                try (JsonGenerator g = factory.createGenerator(buf)) {
                                    g.copyCurrentStructure(p);
                                }
                                reqBytes = buf.toByteArray();
                            }
                            case "expected_approved" -> approved = p.nextBooleanValue();
                            default -> {
                                p.nextToken();
                                p.skipChildren();
                            }
                        }
                    }
                    if (reqBytes == null || approved == null) {
                        throw new IOException("entry missing 'request' or 'expected_approved'");
                    }

                    // Run the production parse path once; copy out only what we keep.
                    RequestVectorParser.State st = parser.acquire();
                    try {
                        parser.vectorize(new ByteArrayInputStream(reqBytes), st);
                        if (n == cap) {
                            cap <<= 1;
                            expected = Arrays.copyOf(expected, cap);
                            if (queries != null) queries = Arrays.copyOf(queries, cap);
                            if (vectors != null) vectors = Arrays.copyOf(vectors, cap);
                            if (json != null) json = Arrays.copyOf(json, cap);
                        }
                        expected[n] = approved;
                        switch (mode) {
                            case SCAN -> {
                                short[] q = new short[DIMS];
                                ReferenceDataset.quantizeInto(st.qvec, q);
                                queries[n] = q;
                            }
                            case SCORE -> vectors[n] = Arrays.copyOf(st.qvec, DIMS);
                            case FULL -> json[n] = reqBytes;
                        }
                        n++;
                    } finally {
                        parser.release(st);
                    }

                    if (limit > 0 && n >= limit) {
                        return new Corpus(n, Arrays.copyOf(expected, n),
                                queries == null ? null : Arrays.copyOf(queries, n),
                                vectors == null ? null : Arrays.copyOf(vectors, n),
                                json == null ? null : Arrays.copyOf(json, n));
                    }
                }
            }
        }
        return new Corpus(n, Arrays.copyOf(expected, n),
                queries == null ? null : Arrays.copyOf(queries, n),
                vectors == null ? null : Arrays.copyOf(vectors, n),
                json == null ? null : Arrays.copyOf(json, n));
    }

    // ------------------------------------------------------------- correctness

    private static final class Confusion {
        long tp, tn, fp, fn;

        long total() {
            return tp + tn + fp + fn;
        }
    }

    /** Single-threaded pass tallying TP/TN/FP/FN so speed work can't silently break detection. */
    private Confusion correctnessPass(IvfFraudScorer scorer, Corpus c, Mode mode) throws IOException {
        Confusion cm = new Confusion();
        RequestVectorParser parser = (mode == Mode.FULL)
                ? new RequestVectorParser(scorer.vectorizer()) : null;
        RequestVectorParser.State st = (parser != null) ? parser.acquire() : null;
        try {
            for (int i = 0; i < c.n; i++) {
                boolean approved = approvedFor(scorer, c, mode, i, parser, st);
                boolean expectFraud = !c.expectedApproved[i];
                if (expectFraud) {
                    if (!approved) cm.tp++; else cm.fn++;
                } else {
                    if (approved) cm.tn++; else cm.fp++;
                }
            }
        } finally {
            if (parser != null) parser.release(st);
        }
        return cm;
    }

    // ------------------------------------------------------------------ timing

    /**
     * Runs {@code passes} over the corpus across {@code threads} workers, recording
     * per-op latency into {@code latencies} (when non-null; {@code null} = untimed
     * warmup). Op-indices are claimed in {@link #CLAIM}-sized chunks to keep the
     * shared cursor off the per-op path. {@code useVirtual} swaps the workers
     * between platform and virtual threads for an apples-to-apples comparison;
     * note the IVF scan never yields, so virtual workers share the (cgroup-pinned)
     * carrier pool and tend to serialize on this CPU-bound path.
     */
    private void runTimed(IvfFraudScorer scorer, Corpus c, Mode mode, int threads,
                          boolean useVirtual, int passes, long[] latencies) {
        final int n = c.n;
        final long totalOps = (long) passes * n;
        final AtomicInteger cursor = new AtomicInteger(0);
        Thread[] pool = new Thread[threads];
        long[] sinks = new long[threads];

        for (int t = 0; t < threads; t++) {
            final int tid = t;
            Runnable work = () -> {
                // Each worker carries its own FULL-mode parser state from the pool.
                RequestVectorParser parser = (mode == Mode.FULL)
                        ? new RequestVectorParser(scorer.vectorizer()) : null;
                RequestVectorParser.State st = (parser != null) ? parser.acquire() : null;
                long sink = 0;
                try {
                    int base;
                    while ((base = cursor.getAndAdd(CLAIM)) < totalOps) {
                        int end = (int) Math.min(base + CLAIM, totalOps);
                        for (int op = base; op < end; op++) {
                            int i = op % n;
                            long s = System.nanoTime();
                            sink += scoreOnce(scorer, c, mode, i, parser, st);
                            long e = System.nanoTime();
                            if (latencies != null) {
                                latencies[op] = e - s;
                            }
                        }
                    }
                } catch (IOException ex) {
                    throw new RuntimeException(ex);
                } finally {
                    if (parser != null) parser.release(st);
                }
                sinks[tid] = sink;
            };
            Thread.Builder builder = useVirtual
                    ? Thread.ofVirtual().name("bench-v" + t)
                    : Thread.ofPlatform().name("bench-" + t);
            pool[t] = builder.unstarted(work);
        }

        for (Thread th : pool) th.start();
        long combined = 0;
        for (Thread th : pool) {
            try {
                th.join();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        for (long s : sinks) combined += s;
        blackhole += combined;
    }

    /** Executes one scored op for entry {@code i}, returning an int to feed the blackhole. */
    private static long scoreOnce(IvfFraudScorer scorer, Corpus c, Mode mode, int i,
                                  RequestVectorParser parser, RequestVectorParser.State st)
            throws IOException {
        return switch (mode) {
            case SCAN -> scorer.countFraudsAmongNearest(c.queries[i]);
            case SCORE -> scorer.scoreVector(c.vectors[i]).approved() ? 1 : 0;
            case FULL -> {
                parser.vectorize(new ByteArrayInputStream(c.requestJson[i]), st);
                yield scorer.scoreVector(st.qvec).approved() ? 1 : 0;
            }
        };
    }

    /** Same as {@link #scoreOnce} but returns the approve verdict for the correctness tally. */
    private static boolean approvedFor(IvfFraudScorer scorer, Corpus c, Mode mode, int i,
                                       RequestVectorParser parser, RequestVectorParser.State st)
            throws IOException {
        return switch (mode) {
            case SCAN -> {
                int frauds = scorer.countFraudsAmongNearest(c.queries[i]);
                yield (double) frauds / IvfFraudScorer.K < IvfFraudScorer.THRESHOLD;
            }
            case SCORE -> scorer.scoreVector(c.vectors[i]).approved();
            case FULL -> {
                parser.vectorize(new ByteArrayInputStream(c.requestJson[i]), st);
                yield scorer.scoreVector(st.qvec).approved();
            }
        };
    }

    // ------------------------------------------------------------------ report

    private void report(long[] latencies, long wallNs, Confusion cm, int n) {
        long ops = latencies.length;
        double wallSec = wallNs / 1e9;
        double throughput = ops / wallSec;

        long[] sorted = latencies.clone();
        Arrays.sort(sorted);
        double meanUs = mean(sorted) / 1_000.0;

        System.out.println("--- latency (per op) ---");
        System.out.printf("  count     : %d ops in %.3f s%n", ops, wallSec);
        System.out.printf("  throughput: %,.0f ops/s%n", throughput);
        System.out.printf("  mean      : %8.2f us%n", meanUs);
        System.out.printf("  p50       : %8.2f us%n", pct(sorted, 50.0) / 1_000.0);
        System.out.printf("  p90       : %8.2f us%n", pct(sorted, 90.0) / 1_000.0);
        System.out.printf("  p99       : %8.2f us%n", pct(sorted, 99.0) / 1_000.0);
        System.out.printf("  p99.9     : %8.2f us%n", pct(sorted, 99.9) / 1_000.0);
        System.out.printf("  max       : %8.2f us%n", sorted[sorted.length - 1] / 1_000.0);
        System.out.println();

        long det = cm.tp + cm.tn;
        double acc = cm.total() == 0 ? 0 : (double) det / cm.total();
        System.out.println("--- detection (vs expected_approved) ---");
        System.out.printf("  TP=%d  TN=%d  FP=%d  FN=%d%n", cm.tp, cm.tn, cm.fp, cm.fn);
        System.out.printf("  accuracy  : %.4f  (FP+FN=%d of %d)%n", acc, cm.fp + cm.fn, cm.total());
        // Touch the blackhole so the JIT/AOT cannot prove the scan is unused.
        if (blackhole == Long.MIN_VALUE) {
            System.out.println("  (blackhole)");
        }
    }

    private void reportThrottle(CpuStat before, CpuStat after) {
        if (!before.valid() || !after.valid()) {
            return;
        }
        long periods = after.nrPeriods - before.nrPeriods;
        long throttled = after.nrThrottled - before.nrThrottled;
        long stallUsec = after.throttledUsec - before.throttledUsec;
        System.out.println();
        System.out.println("--- cfs throttling (timed passes) ---");
        if (periods <= 0) {
            System.out.println("  no CFS periods elapsed (no quota set, or run too short)");
            return;
        }
        double wallPct = 100.0 * throttled / periods;
        System.out.printf("  cfs periods   : %d%n", periods);
        System.out.printf("  throttled     : %d  (%.1f%% of periods hit the quota wall)%n",
                throttled, wallPct);
        System.out.printf("  throttled time: %.1f ms total%n", stallUsec / 1000.0);
        if (throttled > 0) {
            System.out.printf("  avg stall     : %.2f ms per throttled period%n",
                    (stallUsec / 1000.0) / throttled);
        }
    }

    /** Snapshot of cgroup CFS bandwidth counters; {@link #valid} is false off-cgroup. */
    private static final class CpuStat {
        static final CpuStat NONE = new CpuStat(0, 0, 0, false, "");
        final long nrPeriods;
        final long nrThrottled;
        final long throttledUsec;
        final boolean valid;
        final String source;

        CpuStat(long nrPeriods, long nrThrottled, long throttledUsec, boolean valid, String source) {
            this.nrPeriods = nrPeriods;
            this.nrThrottled = nrThrottled;
            this.throttledUsec = throttledUsec;
            this.valid = valid;
            this.source = source;
        }

        boolean valid() {
            return valid;
        }
    }

    /**
     * Reads the cgroup {@code cpu.stat} throttle counters. Tries cgroup v2 first
     * ({@code /sys/fs/cgroup/cpu.stat}, {@code throttled_usec} in microseconds) then the
     * common v1 locations ({@code throttled_time} in nanoseconds). Returns {@link CpuStat#NONE}
     * when no readable cpu.stat is found (e.g. on macOS hosts or outside a CPU-limited cgroup).
     */
    private static CpuStat readCpuStat() {
        String[] candidates = {
                "/sys/fs/cgroup/cpu.stat",            // cgroup v2 (unified)
                "/sys/fs/cgroup/cpu/cpu.stat",        // cgroup v1
                "/sys/fs/cgroup/cpu,cpuacct/cpu.stat" // cgroup v1 (combined controller)
        };
        for (String path : candidates) {
            Path p = Path.of(path);
            if (!Files.isReadable(p)) {
                continue;
            }
            try {
                long nrPeriods = -1, nrThrottled = -1, throttledUsec = -1;
                for (String line : Files.readAllLines(p)) {
                    int sp = line.indexOf(' ');
                    if (sp <= 0) {
                        continue;
                    }
                    String key = line.substring(0, sp);
                    long val;
                    try {
                        val = Long.parseLong(line.substring(sp + 1).trim());
                    } catch (NumberFormatException e) {
                        continue;
                    }
                    switch (key) {
                        case "nr_periods" -> nrPeriods = val;
                        case "nr_throttled" -> nrThrottled = val;
                        case "throttled_usec" -> throttledUsec = val;        // v2: microseconds
                        case "throttled_time" -> throttledUsec = val / 1000; // v1: nanoseconds
                        default -> { }
                    }
                }
                if (nrPeriods >= 0 && nrThrottled >= 0 && throttledUsec >= 0) {
                    String src = path.contains("cpu.stat") && path.equals("/sys/fs/cgroup/cpu.stat")
                            ? "cgroup v2" : "cgroup v1";
                    return new CpuStat(nrPeriods, nrThrottled, throttledUsec, true, src);
                }
            } catch (IOException ignored) {
                // try next candidate
            }
        }
        return CpuStat.NONE;
    }

    private static double mean(long[] a) {
        long sum = 0;
        for (long v : a) sum += v;
        return a.length == 0 ? 0 : (double) sum / a.length;
    }

    /** Nearest-rank percentile over an ascending-sorted array (nanos). */
    private static long pct(long[] sorted, double p) {
        if (sorted.length == 0) return 0;
        int rank = (int) Math.ceil(p / 100.0 * sorted.length) - 1;
        if (rank < 0) rank = 0;
        if (rank >= sorted.length) rank = sorted.length - 1;
        return sorted[rank];
    }

    // -------------------------------------------------------------------- env

    private static Mode parseMode(String s) {
        return switch (s.trim().toLowerCase()) {
            case "scan" -> Mode.SCAN;
            case "score" -> Mode.SCORE;
            case "full" -> Mode.FULL;
            default -> throw new IllegalArgumentException(
                    "BENCH_MODE must be scan|score|full, got: " + s);
        };
    }

    private static String env(String key, String def) {
        String v = System.getenv(key);
        return (v != null && !v.isBlank()) ? v.trim() : def;
    }

    private static int envInt(String key, int def) {
        String v = System.getenv(key);
        if (v != null && !v.isBlank()) {
            try {
                return Integer.parseInt(v.trim());
            } catch (NumberFormatException ignored) {
                // fall through to default
            }
        }
        return def;
    }
}
