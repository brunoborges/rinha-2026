package io.github.brunoborges.rinha2026;

import io.github.brunoborges.rinha2026.kdport.Dataset;
import io.github.brunoborges.rinha2026.kdport.KdTree;
import io.github.brunoborges.rinha2026.kdport.KdTreeBuilder;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies {@link KdTreeFraudScorer} maps the KdTree's top-5 fraud count to the
 * contest decision (denied iff fraud_score &ge; 0.6, i.e. &ge; 3 of 5 neighbors are
 * fraud) and agrees with a brute-force exact top-5 over a random set.
 */
class KdTreeFraudScorerTest {

    private static final int DIMS = 14;

    private static KdTreeFraudScorer build(float[][] pts, boolean[] fraud) {
        int n = pts.length;
        float[] vectors = new float[n * Dataset.STRIDE];
        for (int i = 0; i < n; i++) {
            System.arraycopy(pts[i], 0, vectors, i * Dataset.STRIDE, DIMS);
        }
        KdTree tree = KdTreeBuilder.build(new Dataset(n, vectors, fraud));
        return new KdTreeFraudScorer(new TransactionVectorizer(), tree);
    }

    @Test
    void allNearestFraudIsDenied() {
        // 5 fraud points clustered near the origin, 50 legit points pushed far away.
        int n = 55;
        float[][] pts = new float[n][DIMS];
        boolean[] fraud = new boolean[n];
        for (int i = 0; i < 5; i++) {
            for (int d = 0; d < DIMS; d++) {
                pts[i][d] = 0.001f * i;
            }
            fraud[i] = true;
        }
        for (int i = 5; i < n; i++) {
            for (int d = 0; d < DIMS; d++) {
                pts[i][d] = 0.5f + 0.001f * i;
            }
            fraud[i] = false;
        }
        KdTreeFraudScorer scorer = build(pts, fraud);
        FraudResponse r = scorer.scoreVector(new double[DIMS]); // query at origin
        assertEquals(1.0, r.fraudScore(), 1e-9, "all 5 nearest are fraud");
        assertFalse(r.approved(), "fraud_score 1.0 must be denied");
    }

    @Test
    void allNearestLegitIsApproved() {
        int n = 55;
        float[][] pts = new float[n][DIMS];
        boolean[] fraud = new boolean[n];
        for (int i = 0; i < 5; i++) {
            for (int d = 0; d < DIMS; d++) {
                pts[i][d] = 0.001f * i;
            }
            fraud[i] = false;
        }
        for (int i = 5; i < n; i++) {
            for (int d = 0; d < DIMS; d++) {
                pts[i][d] = 0.5f + 0.001f * i;
            }
            fraud[i] = true;
        }
        KdTreeFraudScorer scorer = build(pts, fraud);
        FraudResponse r = scorer.scoreVector(new double[DIMS]);
        assertEquals(0.0, r.fraudScore(), 1e-9, "all 5 nearest are legit");
        assertTrue(r.approved(), "fraud_score 0.0 must be approved");
    }

    @Test
    void agreesWithBruteForceOnRandomSet() {
        Random rng = new Random(7);
        int n = 1500;
        float[][] pts = new float[n][DIMS];
        boolean[] fraud = new boolean[n];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIMS; d++) {
                pts[i][d] = (rng.nextFloat() * 2f - 1f);
            }
            fraud[i] = rng.nextInt(3) == 0;
        }
        KdTreeFraudScorer scorer = build(pts, fraud);

        for (int t = 0; t < 500; t++) {
            double[] query = new double[DIMS];
            for (int d = 0; d < DIMS; d++) {
                query[d] = rng.nextFloat() * 2f - 1f;
            }
            int expected = bruteForceTop5Frauds(pts, fraud, query);
            if (expected < 0) {
                continue; // ambiguous 5th/6th tie: tie-break order is impl-defined, skip
            }
            double expectedScore = (double) expected / IvfFraudScorer.K;
            FraudResponse r = scorer.scoreVector(query);
            assertEquals(expectedScore < IvfFraudScorer.THRESHOLD, r.approved(),
                    "decision must match brute-force exact top-5 for query " + t);
        }
    }

    /** Returns the fraud count among the exact top-5, or -1 if the 5th/6th neighbor distances tie. */
    private static int bruteForceTop5Frauds(float[][] pts, boolean[] fraud, double[] query) {
        // Distance in the same i16-quantized space the tree uses (round(v*10000)).
        int n = pts.length;
        long[] dist = new long[n];
        Integer[] idx = new Integer[n];
        short[] qq = new short[DIMS];
        for (int d = 0; d < DIMS; d++) {
            qq[d] = quantize(query[d]);
        }
        for (int i = 0; i < n; i++) {
            long s = 0;
            for (int d = 0; d < DIMS; d++) {
                int diff = quantize(pts[i][d]) - qq[d];
                s += (long) diff * diff;
            }
            dist[i] = s;
            idx[i] = i;
        }
        java.util.Arrays.sort(idx, (a, b) -> {
            int c = Long.compare(dist[a], dist[b]);
            return c != 0 ? c : Integer.compare(a, b);
        });
        int k = Math.min(IvfFraudScorer.K, n);
        if (n > k && dist[idx[k - 1]] == dist[idx[k]]) {
            return -1; // boundary tie
        }
        int frauds = 0;
        for (int i = 0; i < k; i++) {
            if (fraud[idx[i]]) {
                frauds++;
            }
        }
        return frauds;
    }

    private static short quantize(double v) {
        if (v <= -1.0) {
            return (short) -10000;
        }
        if (v >= 1.0) {
            return (short) 10000;
        }
        return (short) Math.round(v * 10000);
    }
}
