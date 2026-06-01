package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class TransactionVectorizerTest {

    private static final ObjectMapper MAPPER = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

    private static final double EPS = 1e-4;

    private final TransactionVectorizer vectorizer = new TransactionVectorizer();

    @Test
    void vectorizesLegitExampleFromDetectionRules() throws Exception {
        String payload = """
                {
                  "id": "tx-1329056812",
                  "transaction": { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
                  "customer": { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
                  "merchant": { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
                  "terminal": { "is_online": false, "card_present": true, "km_from_home": 29.23 },
                  "last_transaction": null
                }
                """;

        double[] expected = {0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006};

        double[] actual = vectorizer.vectorize(parse(payload));

        assertArrayEquals(expected, actual, EPS);
    }

    @Test
    void vectorizesFraudExampleFromDetectionRules() throws Exception {
        String payload = """
                {
                  "id": "tx-3330991687",
                  "transaction": { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
                  "customer": { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
                  "merchant": { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
                  "terminal": { "is_online": false, "card_present": true, "km_from_home": 952.27 },
                  "last_transaction": null
                }
                """;

        double[] expected = {0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055};

        double[] actual = vectorizer.vectorize(parse(payload));

        assertArrayEquals(expected, actual, EPS);
    }

    @Test
    void normalizesMinutesAndKmWhenLastTransactionPresent() throws Exception {
        String payload = """
                {
                  "id": "tx-1",
                  "transaction": { "amount": 100, "installments": 1, "requested_at": "2026-03-11T12:00:00Z" },
                  "customer": { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
                  "merchant": { "id": "MERC-001", "mcc": "5411", "avg_amount": 100 },
                  "terminal": { "is_online": true, "card_present": false, "km_from_home": 10 },
                  "last_transaction": { "timestamp": "2026-03-11T11:00:00Z", "km_from_current": 500 }
                }
                """;

        double[] actual = vectorizer.vectorize(parse(payload));

        assertEquals(60.0 / 1440.0, actual[5], EPS);
        assertEquals(0.5, actual[6], EPS);
        assertEquals(1.0, actual[9], EPS);
        assertEquals(0.0, actual[10], EPS);
    }

    @Test
    void usesDefaultRiskForUnknownMcc() throws Exception {
        String payload = """
                {
                  "id": "tx-1",
                  "transaction": { "amount": 100, "installments": 1, "requested_at": "2026-03-11T12:00:00Z" },
                  "customer": { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": [] },
                  "merchant": { "id": "MERC-999", "mcc": "0000", "avg_amount": 100 },
                  "terminal": { "is_online": false, "card_present": true, "km_from_home": 10 },
                  "last_transaction": null
                }
                """;

        double[] actual = vectorizer.vectorize(parse(payload));

        assertEquals(TransactionVectorizer.DEFAULT_MCC_RISK, actual[12], EPS);
        assertEquals(1.0, actual[11], EPS);
    }

    @Test
    void clampsValuesOutsideUnitInterval() {
        assertEquals(0.0, TransactionVectorizer.clamp(-5.0), EPS);
        assertEquals(1.0, TransactionVectorizer.clamp(42.0), EPS);
        assertEquals(0.3, TransactionVectorizer.clamp(0.3), EPS);
    }

    private static FraudRequest parse(String json) throws Exception {
        return MAPPER.readValue(json, FraudRequest.class);
    }
}
