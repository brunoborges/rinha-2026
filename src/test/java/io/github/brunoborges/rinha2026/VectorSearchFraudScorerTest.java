package io.github.brunoborges.rinha2026;

import org.junit.jupiter.api.Test;

import java.time.Instant;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VectorSearchFraudScorerTest {

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private final TransactionVectorizer vectorizer = new TransactionVectorizer();

    private static final FraudRequest SAMPLE = new FraudRequest(
            "tx-1",
            new FraudRequest.Transaction(100.0, 1, Instant.parse("2026-03-11T12:00:00Z")),
            new FraudRequest.Customer(100.0, 1, List.of("MERC-001")),
            new FraudRequest.Merchant("MERC-001", "5411", 100.0),
            new FraudRequest.Terminal(false, true, 10.0),
            null);

    @Test
    void scoresFraudWhenMajorityOfNearestAreFraud() {
        double[] q = vectorizer.vectorize(SAMPLE);
        // 3 fraud + 2 legit are the nearest; the rest are far away.
        ReferenceDataset dataset = datasetAround(q,
                new boolean[]{true, true, true, false, false},
                new boolean[]{false, false});

        FraudResponse response = new VectorSearchFraudScorer(vectorizer, dataset).score(SAMPLE);

        assertEquals(0.6, response.fraudScore(), 1e-9);
        assertFalse(response.approved());
    }

    @Test
    void approvesWhenMajorityOfNearestAreLegit() {
        double[] q = vectorizer.vectorize(SAMPLE);
        // 2 fraud + 3 legit are the nearest; far records are all fraud and ignored.
        ReferenceDataset dataset = datasetAround(q,
                new boolean[]{true, true, false, false, false},
                new boolean[]{true, true, true});

        FraudResponse response = new VectorSearchFraudScorer(vectorizer, dataset).score(SAMPLE);

        assertEquals(0.4, response.fraudScore(), 1e-9);
        assertTrue(response.approved());
    }

    @Test
    void handlesDatasetSmallerThanK() {
        double[] q = vectorizer.vectorize(SAMPLE);
        ReferenceDataset dataset = datasetAround(q, new boolean[]{true, false}, new boolean[]{});

        FraudResponse response = new VectorSearchFraudScorer(vectorizer, dataset).score(SAMPLE);

        // 1 fraud of 2 records -> 0.5, which is < 0.6 -> approved.
        assertEquals(0.5, response.fraudScore(), 1e-9);
        assertTrue(response.approved());
    }

    /**
     * Builds a dataset whose first records sit a hair away from {@code q} (the
     * guaranteed nearest neighbors, in order) followed by records placed far
     * away (must be ignored by the top-K search).
     */
    private static ReferenceDataset datasetAround(double[] q, boolean[] near, boolean[] far) {
        int total = near.length + far.length;
        float[] vectors = new float[total * DIMS];
        boolean[] fraud = new boolean[total];

        int row = 0;
        for (int n = 0; n < near.length; n++, row++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[row * DIMS + d] = (float) q[d];
            }
            // Distinct, tiny, increasing offsets so ordering is deterministic.
            vectors[row * DIMS] = (float) (q[0] + (n + 1) * 1e-4);
            fraud[row] = near[n];
        }
        for (int f = 0; f < far.length; f++, row++) {
            for (int d = 0; d < DIMS; d++) {
                vectors[row * DIMS + d] = (float) q[d];
            }
            vectors[row * DIMS] = (float) (q[0] + 100.0 + f);
            fraud[row] = far[f];
        }
        return ReferenceDataset.of(vectors, fraud, total);
    }
}
