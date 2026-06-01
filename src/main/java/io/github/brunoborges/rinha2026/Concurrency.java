package io.github.brunoborges.rinha2026;

import java.util.logging.Logger;

/**
 * Sizing for the per-request working-memory pools (scorer {@code Scratch} and
 * {@link RequestVectorParser} {@code State}).
 *
 * <p>These pools bound how many requests can be parsed/scanned concurrently per
 * instance. They were previously sized to {@link Runtime#availableProcessors()},
 * but under the {@code cpus=0.45} cgroup limit that value is always {@code 1}
 * (the quota rounds up to one), which serialized scoring even though the IVF
 * scan is memory-latency-bound and leaves the CPU mostly idle while stalled on
 * the mmap'd dataset. Allowing a few concurrent scans overlaps those stalls and
 * raises throughput without exceeding the CPU quota.
 *
 * <p>The {@code WORKERS} environment variable tunes this at runtime (no rebuild),
 * mirroring the {@code NPROBE}/{@code SCAN_CAP} knobs.
 */
final class Concurrency {

    private static final Logger LOG = Logger.getLogger(Concurrency.class.getName());

    /** Default concurrent workers per instance when {@code WORKERS} is unset. */
    static final int DEFAULT_WORKERS = 4;

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
