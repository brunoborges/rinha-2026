package io.github.brunoborges.rinha2026;

import java.util.logging.Logger;

/**
 * Sizing for the per-request working-memory pools (scorer {@code Scratch} and
 * {@link RequestVectorParser} {@code State}).
 *
 * <p>These pools bound how many requests can be parsed/scanned concurrently per
 * instance. The default is a single worker: under the sub-1-CPU cgroup quota the
 * IVF scan is the sole CPU-bound stage and already saturates the core, so adding
 * workers cannot raise throughput &mdash; it only oversubscribes the one core and
 * drives CFS throttling to 100% of periods, blowing the latency tail up by orders
 * of magnitude (p99 ~0.7ms single-threaded vs ~60ms+ at 3+ workers) while p50 and
 * throughput stay flat. Since the load is scored on p99, a single thread wins.
 *
 * <p>The {@code WORKERS} environment variable tunes this at runtime (no rebuild),
 * mirroring the {@code NPROBE}/{@code SCAN_CAP} knobs.
 */
final class Concurrency {

    private static final Logger LOG = Logger.getLogger(Concurrency.class.getName());

    /** Default concurrent workers per instance when {@code WORKERS} is unset. */
    static final int DEFAULT_WORKERS = 1;

    private Concurrency() {
    }

    /**
     * Number of concurrent workers (pool size) per instance, from the
     * {@code WORKERS} environment variable, clamped to at least {@code 1}.
     */
    static int workers() {
        String env = System.getenv("WORKERS");
        if (env != null && !env.isBlank()) {
            try {
                return Math.max(1, Integer.parseInt(env.trim()));
            } catch (NumberFormatException e) {
                LOG.warning(() -> "Invalid WORKERS '" + env + "'; using default " + DEFAULT_WORKERS);
            }
        }
        return DEFAULT_WORKERS;
    }
}
