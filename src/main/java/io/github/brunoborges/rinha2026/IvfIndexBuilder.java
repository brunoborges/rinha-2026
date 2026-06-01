package io.github.brunoborges.rinha2026;

import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;

/**
 * Offline builder for the IVF (inverted-file) index baked into the version-2
 * {@code .bin} (see {@link ReferenceDataset}). It clusters the quantized
 * reference vectors with k-means, then reorders them so each cluster is
 * contiguous, letting a query scan only a few probed clusters instead of the
 * whole dataset.
 *
 * <p>This runs once at image-build time on the (unconstrained) build host, so it
 * parallelizes the expensive nearest-centroid passes. Centroids are trained in
 * {@code float} and quantized to {@code int16} only when serialized, per the
 * usual k-means-quality guidance. To keep build time bounded, k-means trains on
 * a random subsample and then does a single full assignment of every record.
 */
public final class IvfIndexBuilder {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;

    /** Default number of clusters; overridable so build cost/recall can be tuned. */
    public static final int DEFAULT_CLUSTERS = 2048;

    /** Default Lloyd iterations over the training subsample. */
    public static final int DEFAULT_ITERATIONS = 10;

    /** Fixed seed so builds are deterministic. */
    public static final long SEED = 0x5DEECE66DL;

    private IvfIndexBuilder() {
    }

    /** The reordered vectors/labels plus the centroid table and CSR offsets. */
    public record Result(int count, int clusters, short[] vectors, byte[] labels,
                         short[] centroids, int[] offsets) {
    }

    /** Builds an index with default parameters. */
    public static Result build(ReferenceDataset source) {
        return build(source, DEFAULT_CLUSTERS, DEFAULT_ITERATIONS);
    }

    /**
     * Builds an IVF index from {@code source}.
     *
     * @param source     the loaded (unindexed) dataset
     * @param clusters   requested number of clusters (clamped to {@code [1, count]})
     * @param iterations Lloyd iterations over the training subsample
     */
    public static Result build(ReferenceDataset source, int clusters, int iterations) {
        int count = source.count();
        if (count == 0) {
            throw new IllegalArgumentException("cannot build an index over an empty dataset");
        }
        clusters = Math.max(1, Math.min(clusters, count));

        short[] vectors = source.toQuantizedArray();
        byte[] labels = source.toLabelArray();

        int sampleSize = (int) Math.min(count,
                Math.min(200_000L, Math.max(100_000L, (long) clusters * 50)));
        Random random = new Random(SEED);

        float[] sample = sample(vectors, count, sampleSize, random);
        float[] centroids = kmeansPlusPlusInit(sample, sampleSize, clusters, random);
        lloyd(sample, sampleSize, centroids, clusters, iterations, random);

        int[] assignment = assignAll(vectors, count, centroids, clusters);

        int[] sizes = new int[clusters];
        for (int c : assignment) {
            sizes[c]++;
        }
        int[] offsets = new int[clusters + 1];
        for (int c = 0; c < clusters; c++) {
            offsets[c + 1] = offsets[c] + sizes[c];
        }

        short[] reorderedVectors = new short[count * DIMS];
        byte[] reorderedLabels = new byte[count];
        int[] cursor = Arrays.copyOf(offsets, clusters);
        for (int i = 0; i < count; i++) {
            int c = assignment[i];
            int pos = cursor[c]++;
            System.arraycopy(vectors, i * DIMS, reorderedVectors, pos * DIMS, DIMS);
            reorderedLabels[pos] = labels[i];
        }

        short[] centroidShorts = quantizeCentroids(centroids, clusters);
        logStats(sizes);

        return new Result(count, clusters, reorderedVectors, reorderedLabels, centroidShorts, offsets);
    }

    /** Materializes {@code sampleSize} random records as a flat {@code float[]}. */
    private static float[] sample(short[] vectors, int count, int sampleSize, Random random) {
        float[] sample = new float[sampleSize * DIMS];
        for (int s = 0; s < sampleSize; s++) {
            int src = random.nextInt(count) * DIMS;
            int dst = s * DIMS;
            for (int d = 0; d < DIMS; d++) {
                sample[dst + d] = vectors[src + d];
            }
        }
        return sample;
    }

    /** k-means++ seeding over the training sample. */
    private static float[] kmeansPlusPlusInit(float[] sample, int sampleSize, int clusters, Random random) {
        float[] centroids = new float[clusters * DIMS];
        int first = random.nextInt(sampleSize);
        System.arraycopy(sample, first * DIMS, centroids, 0, DIMS);

        double[] nearest = new double[sampleSize];
        Arrays.fill(nearest, Double.MAX_VALUE);

        for (int c = 1; c < clusters; c++) {
            int prev = c - 1;
            double total = 0.0;
            for (int s = 0; s < sampleSize; s++) {
                double d = squared(sample, s * DIMS, centroids, prev * DIMS);
                if (d < nearest[s]) {
                    nearest[s] = d;
                }
                total += nearest[s];
            }
            int chosen = pickWeighted(nearest, total, sampleSize, random);
            System.arraycopy(sample, chosen * DIMS, centroids, c * DIMS, DIMS);
        }
        return centroids;
    }

    private static int pickWeighted(double[] weights, double total, int n, Random random) {
        if (total <= 0.0) {
            return random.nextInt(n);
        }
        double target = random.nextDouble() * total;
        double acc = 0.0;
        for (int s = 0; s < n; s++) {
            acc += weights[s];
            if (acc >= target) {
                return s;
            }
        }
        return n - 1;
    }

    /** Lloyd's algorithm over the training sample; reseeds empty clusters. */
    private static void lloyd(float[] sample, int sampleSize, float[] centroids, int clusters,
                              int iterations, Random random) {
        for (int it = 0; it < iterations; it++) {
            int[] assign = IntStream.range(0, sampleSize).parallel()
                    .map(s -> nearest(sample, s * DIMS, centroids, clusters))
                    .toArray();

            double[] sums = new double[clusters * DIMS];
            int[] counts = new int[clusters];
            for (int s = 0; s < sampleSize; s++) {
                int c = assign[s];
                counts[c]++;
                int base = c * DIMS;
                int src = s * DIMS;
                for (int d = 0; d < DIMS; d++) {
                    sums[base + d] += sample[src + d];
                }
            }
            for (int c = 0; c < clusters; c++) {
                int base = c * DIMS;
                if (counts[c] == 0) {
                    int reseed = random.nextInt(sampleSize) * DIMS;
                    System.arraycopy(sample, reseed, centroids, base, DIMS);
                } else {
                    double inv = 1.0 / counts[c];
                    for (int d = 0; d < DIMS; d++) {
                        centroids[base + d] = (float) (sums[base + d] * inv);
                    }
                }
            }
        }
    }

    /** Assigns every record to its nearest centroid, in parallel. */
    private static int[] assignAll(short[] vectors, int count, float[] centroids, int clusters) {
        return IntStream.range(0, count).parallel()
                .map(i -> nearestShort(vectors, i * DIMS, centroids, clusters))
                .toArray();
    }

    private static int nearest(float[] points, int offset, float[] centroids, int clusters) {
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < clusters; c++) {
            double d = squared(points, offset, centroids, c * DIMS);
            if (d < bestDist) {
                bestDist = d;
                best = c;
            }
        }
        return best;
    }

    private static int nearestShort(short[] points, int offset, float[] centroids, int clusters) {
        int best = 0;
        double bestDist = Double.MAX_VALUE;
        for (int c = 0; c < clusters; c++) {
            int cbase = c * DIMS;
            double sum = 0.0;
            for (int d = 0; d < DIMS; d++) {
                double diff = points[offset + d] - centroids[cbase + d];
                sum += diff * diff;
            }
            if (sum < bestDist) {
                bestDist = sum;
                best = c;
            }
        }
        return best;
    }

    private static double squared(float[] a, int aOff, float[] b, int bOff) {
        double sum = 0.0;
        for (int d = 0; d < DIMS; d++) {
            double diff = a[aOff + d] - b[bOff + d];
            sum += diff * diff;
        }
        return sum;
    }

    private static short[] quantizeCentroids(float[] centroids, int clusters) {
        short[] out = new short[clusters * DIMS];
        for (int i = 0; i < out.length; i++) {
            long v = Math.round(centroids[i]);
            if (v > Short.MAX_VALUE) {
                v = Short.MAX_VALUE;
            } else if (v < Short.MIN_VALUE) {
                v = Short.MIN_VALUE;
            }
            out[i] = (short) v;
        }
        return out;
    }

    private static void logStats(int[] sizes) {
        int clusters = sizes.length;
        int[] sorted = sizes.clone();
        Arrays.sort(sorted);
        int empty = 0;
        for (int s : sizes) {
            if (s == 0) {
                empty++;
            }
        }
        int p50 = sorted[(int) (clusters * 0.50)];
        int p99 = sorted[Math.min(clusters - 1, (int) (clusters * 0.99))];
        int max = sorted[clusters - 1];
        System.out.printf(
                "IVF: %,d clusters | cluster size p50=%,d p99=%,d max=%,d | empty=%d%n",
                clusters, p50, p99, max, empty);
    }
}
