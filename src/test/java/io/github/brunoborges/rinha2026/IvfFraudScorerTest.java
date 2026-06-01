package io.github.brunoborges.rinha2026;

import org.junit.jupiter.api.Test;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class IvfFraudScorerTest {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private static final int K = 5;

    @Test
    void ivfAgreesWithBruteForceOnClusteredData() throws Exception {
        Random random = new Random(42);
        int blobs = 20;
        int perBlob = 300;
        int count = blobs * perBlob;

        float[] vectors = new float[count * DIMS];
        boolean[] fraud = new boolean[count];
        float[][] centers = new float[blobs][DIMS];
        for (int b = 0; b < blobs; b++) {
            for (int d = 0; d < DIMS; d++) {
                centers[b][d] = random.nextFloat();
            }
        }
        int row = 0;
        for (int b = 0; b < blobs; b++) {
            // Each blob skews fraud/legit so neighborhoods carry signal.
            double fraudProb = (b % 2 == 0) ? 0.1 : 0.85;
            for (int j = 0; j < perBlob; j++, row++) {
                for (int d = 0; d < DIMS; d++) {
                    float v = centers[b][d] + (float) (random.nextGaussian() * 0.01);
                    vectors[row * DIMS + d] = clamp01(v);
                }
                fraud[row] = random.nextDouble() < fraudProb;
            }
        }

        ReferenceDataset source = ReferenceDataset.of(vectors, fraud, count);
        IvfIndexBuilder.Result index = IvfIndexBuilder.build(source, 64, 12);

        Path bin = Files.createTempFile("ivf", ".bin");
        try {
            ReferenceDataset.writeBinaryV2(bin, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());
            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                assertTrue(mapped.hasIndex());
                assertEquals(64, mapped.clusters());
                assertEquals(count, mapped.count());

                // Generous probing should match brute force almost perfectly.
                IvfFraudScorer scorer = new IvfFraudScorer(new TransactionVectorizer(), mapped, 64, count);

                int agree = 0;
                int trials = 400;
                for (int t = 0; t < trials; t++) {
                    short[] q = randomQuery(random);
                    int ivf = scorer.countFraudsAmongNearest(q);
                    int brute = bruteForceFraudCount(mapped, q);
                    boolean ivfApproved = (double) ivf / K < IvfFraudScorer.THRESHOLD;
                    boolean bruteApproved = (double) brute / K < IvfFraudScorer.THRESHOLD;
                    if (ivfApproved == bruteApproved) {
                        agree++;
                    }
                }
                // With nprobe == clusters the search is exhaustive: decisions must match exactly.
                assertEquals(trials, agree, "IVF decisions diverged from brute force");
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void scoreEndToEndApprovesLegitNeighborhood() throws Exception {
        Random random = new Random(7);
        int count = 4000;
        float[] vectors = new float[count * DIMS];
        boolean[] fraud = new boolean[count];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[i * DIMS + d] = random.nextFloat();
            }
            // Make the region near the all-0.5 vector overwhelmingly legit.
            fraud[i] = random.nextDouble() < 0.5;
        }
        // Plant a tight legit cluster around 0.5 so a 0.5 query is approved.
        for (int i = 0; i < 50; i++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[i * DIMS + d] = 0.5f + (float) (random.nextGaussian() * 0.001);
            }
            fraud[i] = false;
        }

        ReferenceDataset source = ReferenceDataset.of(vectors, fraud, count);
        IvfIndexBuilder.Result index = IvfIndexBuilder.build(source, 32, 10);
        Path bin = Files.createTempFile("ivf-e2e", ".bin");
        try {
            ReferenceDataset.writeBinaryV2(bin, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());
            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                IvfFraudScorer scorer = new IvfFraudScorer(new TransactionVectorizer(), mapped, 8, 100_000);
                short[] q = new short[DIMS];
                Arrays.fill(q, ReferenceDataset.quantize(0.5));
                int frauds = scorer.countFraudsAmongNearest(q);
                assertTrue(frauds <= 1, "expected a legit-dominated neighborhood, got frauds=" + frauds);
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void scanCapBoundsWorkAndStillScoresLegit() throws Exception {
        Random random = new Random(11);
        int count = 6000;
        float[] vectors = new float[count * DIMS];
        boolean[] fraud = new boolean[count];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[i * DIMS + d] = random.nextFloat();
            }
            fraud[i] = false;
        }
        ReferenceDataset source = ReferenceDataset.of(vectors, fraud, count);
        IvfIndexBuilder.Result index = IvfIndexBuilder.build(source, 16, 8);
        Path bin = Files.createTempFile("ivf-cap", ".bin");
        try {
            ReferenceDataset.writeBinaryV2(bin, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());
            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                // scanCap below a single cluster still yields valid neighbors (all legit).
                IvfFraudScorer scorer = new IvfFraudScorer(new TransactionVectorizer(), mapped, 4, 10);
                short[] q = randomQuery(random);
                assertEquals(0, scorer.countFraudsAmongNearest(q));
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void rejectsUnindexedDataset() {
        ReferenceDataset source = ReferenceDataset.of(new float[DIMS], new boolean[]{false}, 1);
        assertThrows(IllegalArgumentException.class,
                () -> new IvfFraudScorer(new TransactionVectorizer(), source));
    }

    @Test
    void buildReordersButPreservesNeighborhoods() throws Exception {
        Random random = new Random(99);
        int count = 1500;
        float[] vectors = new float[count * DIMS];
        boolean[] fraud = new boolean[count];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[i * DIMS + d] = random.nextFloat();
            }
            fraud[i] = random.nextBoolean();
        }
        ReferenceDataset source = ReferenceDataset.of(vectors, fraud, count);
        IvfIndexBuilder.Result index = IvfIndexBuilder.build(source, 24, 10);
        Path bin = Files.createTempFile("ivf-order", ".bin");
        try {
            ReferenceDataset.writeBinaryV2(bin, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());
            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                // Full-scan KNN fraud count must be identical on source and reordered set.
                int divergences = 0;
                for (int t = 0; t < 200; t++) {
                    short[] q = randomQuery(random);
                    if (bruteForceFraudCount(source, q) != bruteForceFraudCount(mapped, q)) {
                        divergences++;
                    }
                }
                assertEquals(0, divergences, "reordering must not change the true KNN result");
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    private static short[] randomQuery(Random random) {
        short[] q = new short[DIMS];
        for (int d = 0; d < DIMS; d++) {
            q[d] = ReferenceDataset.quantize(random.nextFloat());
        }
        return q;
    }

    /** Exhaustive top-K fraud count over every record (the oracle). */
    private static int bruteForceFraudCount(ReferenceDataset dataset, short[] query) {
        int count = dataset.count();
        int k = Math.min(K, count);
        long[] bestDist = new long[k];
        boolean[] bestFraud = new boolean[k];
        Arrays.fill(bestDist, Long.MAX_VALUE);
        int worst = 0;
        for (int i = 0; i < count; i++) {
            long d2 = dataset.squaredDistance(query, i);
            if (d2 < bestDist[worst]) {
                bestDist[worst] = d2;
                bestFraud[worst] = dataset.isFraud(i);
                worst = 0;
                for (int j = 1; j < k; j++) {
                    if (bestDist[j] > bestDist[worst]) {
                        worst = j;
                    }
                }
            }
        }
        int frauds = 0;
        for (boolean f : bestFraud) {
            if (f) {
                frauds++;
            }
        }
        return frauds;
    }

    private static float clamp01(float v) {
        if (v < 0f) {
            return 0f;
        }
        return Math.min(v, 1f);
    }
}
