package io.github.brunoborges.rinha2026;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FraudRequestParserTest {

    private static final String FULL = """
            {
              "id": "tx-3576980410",
              "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
              "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
              "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
              "terminal": {"is_online": false, "card_present": true, "km_from_home": 13.7090520965},
              "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774}
            }""";

    private static FraudRequest parse(String json) throws IOException {
        return FraudRequestParser.parse(new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void parsesFullPayload() throws Exception {
        FraudRequest r = parse(FULL);

        assertEquals("tx-3576980410", r.id());
        assertEquals(384.88, r.transaction().amount());
        assertEquals(3, r.transaction().installments());
        assertEquals(Instant.parse("2026-03-11T20:23:35Z"), r.transaction().requestedAt());
        assertEquals(769.76, r.customer().avgAmount());
        assertEquals(3, r.customer().txCount24h());
        assertEquals(2, r.customer().knownMerchants().size());
        assertEquals("MERC-001", r.merchant().id());
        assertEquals("5912", r.merchant().mcc());
        assertEquals(298.95, r.merchant().avgAmount());
        assertEquals(false, r.terminal().isOnline());
        assertEquals(true, r.terminal().cardPresent());
        assertEquals(13.7090520965, r.terminal().kmFromHome());
        assertEquals(Instant.parse("2026-03-11T14:58:35Z"), r.lastTransaction().timestamp());
        assertEquals(18.8626479774, r.lastTransaction().kmFromCurrent());
    }

    @Test
    void preservesNullLastTransaction() throws Exception {
        FraudRequest r = parse(FULL.replaceFirst(
                "(?s)\"last_transaction\": \\{.*?\\}", "\"last_transaction\": null"));

        assertNull(r.lastTransaction());
        assertEquals("tx-3576980410", r.id());
    }

    @Test
    void leavesAbsentSectionsNull() throws Exception {
        FraudRequest r = parse("{\"id\":\"tx-1\"}");

        assertEquals("tx-1", r.id());
        assertNull(r.transaction());
        assertNull(r.customer());
        assertNull(r.merchant());
        assertNull(r.terminal());
        assertNull(r.lastTransaction());
    }

    @Test
    void skipsUnknownFieldsIncludingNestedObjectsAndArrays() throws Exception {
        String json = """
                {
                  "id": "tx-2",
                  "extra_top": {"a": 1, "b": [1, 2, {"c": 3}]},
                  "transaction": {"amount": 10.0, "installments": 1, "requested_at": "2026-01-01T00:00:00Z", "junk": [7]},
                  "tags": ["x", "y"]
                }""";

        FraudRequest r = parse(json);

        assertEquals("tx-2", r.id());
        assertEquals(10.0, r.transaction().amount());
        assertEquals(1, r.transaction().installments());
    }

    @Test
    void rejectsNonObjectRoot() {
        assertThrows(IOException.class, () -> parse("not json"));
        assertThrows(IOException.class, () -> parse("[1,2,3]"));
    }

    @Test
    void rejectsMalformedJson() {
        assertThrows(IOException.class, () -> parse("{\"id\": "));
    }

    @Test
    void rejectsInvalidTimestamp() {
        IOException e = assertThrows(IOException.class,
                () -> parse("{\"transaction\":{\"amount\":1,\"installments\":1,\"requested_at\":\"nope\"}}"));
        assertTrue(e.getMessage().contains("timestamp"));
    }
}
