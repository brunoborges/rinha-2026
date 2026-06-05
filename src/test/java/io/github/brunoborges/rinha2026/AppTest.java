package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Drives the transport-agnostic HTTP pipeline ({@link HttpConnection} + {@link HttpRouter}) directly
 * with raw request bytes, asserting the contract responses. This keeps the suite host-runnable: the
 * Linux-only fd-passing transport ({@link FdEpollServer}/{@code LinuxSyscalls}) is never loaded, so the
 * core parsing/routing/scoring logic is verified off-Linux exactly as it runs in production.
 */
class AppTest {

    private static final ObjectMapper MAPPER = new ObjectMapper();

    private static final String SAMPLE_PAYLOAD = """
            {
              "id": "tx-3576980410",
              "transaction": {
                "amount": 384.88,
                "installments": 3,
                "requested_at": "2026-03-11T20:23:35Z"
              },
              "customer": {
                "avg_amount": 769.76,
                "tx_count_24h": 3,
                "known_merchants": ["MERC-009", "MERC-001", "MERC-001"]
              },
              "merchant": {
                "id": "MERC-001",
                "mcc": "5912",
                "avg_amount": 298.95
              },
              "terminal": {
                "is_online": false,
                "card_present": true,
                "km_from_home": 13.7090520965
              },
              "last_transaction": {
                "timestamp": "2026-03-11T14:58:35Z",
                "km_from_current": 18.8626479774
              }
            }
            """;

    private static HttpRouter router;

    @BeforeAll
    static void buildRouter() {
        router = new HttpRouter(App.loadDefaultScorer());
        router.markReady();
    }

    @Test
    void readyReturns200() {
        assertEquals(HttpResponses.RESP_READY, route("GET", "/ready", null));
    }

    @Test
    void readyRejectsPost() {
        assertEquals(HttpResponses.RESP_METHOD_NOT_ALLOWED, route("POST", "/ready", ""));
    }

    @Test
    void fraudScoreReturnsContract() throws Exception {
        int idx = route("POST", "/fraud-score", SAMPLE_PAYLOAD);
        assertTrue(idx >= HttpResponses.RESP_FRAUD_0 && idx <= HttpResponses.RESP_FRAUD_5,
                "expected a fraud-score response, got index " + idx);
        JsonNode body = MAPPER.readTree(bodyOf(idx));
        assertTrue(body.get("approved").isBoolean());
        assertTrue(body.get("fraud_score").isNumber());
    }

    @Test
    void fraudScoreAcceptsNullLastTransaction() {
        String payload = SAMPLE_PAYLOAD.replaceFirst(
                "(?s)\"last_transaction\": \\{.*?\\}", "\"last_transaction\": null");
        int idx = route("POST", "/fraud-score", payload);
        assertTrue(idx >= HttpResponses.RESP_FRAUD_0 && idx <= HttpResponses.RESP_FRAUD_5,
                "expected a fraud-score response, got index " + idx);
    }

    @Test
    void fraudScoreRejectsMissingSection() {
        assertEquals(HttpResponses.RESP_BAD_REQUEST, route("POST", "/fraud-score", "{\"id\":\"tx-1\"}"));
    }

    @Test
    void fraudScoreRejectsInvalidJson() {
        assertEquals(HttpResponses.RESP_BAD_REQUEST, route("POST", "/fraud-score", "not json"));
    }

    @Test
    void fraudScoreRejectsGet() {
        assertEquals(HttpResponses.RESP_METHOD_NOT_ALLOWED, route("GET", "/fraud-score", null));
    }

    /** Feed a raw HTTP/1.1 request through the parser+router and return the chosen response index. */
    private static int route(String method, String path, String body) {
        byte[] raw = buildRequest(method, path, body);
        HttpConnection conn = new HttpConnection();
        System.arraycopy(raw, 0, conn.buf, 0, raw.length);
        conn.pos = raw.length;
        assertEquals(HttpConnection.READY, conn.tryParse(), "request did not parse to READY");
        return router.responseIndex(conn);
    }

    private static byte[] buildRequest(String method, String path, String body) {
        StringBuilder sb = new StringBuilder();
        sb.append(method).append(' ').append(path).append(" HTTP/1.1\r\n");
        sb.append("Host: test\r\n");
        if (body != null) {
            byte[] bodyBytes = body.getBytes(StandardCharsets.UTF_8);
            sb.append("Content-Type: application/json\r\n");
            sb.append("Content-Length: ").append(bodyBytes.length).append("\r\n");
        }
        sb.append("\r\n");
        if (body != null) {
            sb.append(body);
        }
        return sb.toString().getBytes(StandardCharsets.UTF_8);
    }

    /** Extract the JSON body from a pre-framed response. */
    private static String bodyOf(int idx) {
        byte[] full = HttpResponses.bytes(idx);
        String s = new String(full, StandardCharsets.UTF_8);
        int sep = s.indexOf("\r\n\r\n");
        return s.substring(sep + 4);
    }
}
