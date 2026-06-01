package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

    private static HttpServer server;
    private static HttpClient client;
    private static String baseUrl;

    @BeforeAll
    static void startServer() throws Exception {
        server = new App().start(0);
        baseUrl = "http://localhost:" + server.getAddress().getPort();
        client = HttpClient.newHttpClient();
    }

    @AfterAll
    static void stopServer() {
        server.stop(0);
    }

    @Test
    void readyReturns200() throws Exception {
        HttpResponse<String> response = client.send(
                HttpRequest.newBuilder(URI.create(baseUrl + "/ready")).GET().build(),
                HttpResponse.BodyHandlers.ofString());

        assertEquals(200, response.statusCode());
    }

    @Test
    void readyRejectsPost() throws Exception {
        HttpResponse<String> response = client.send(
                HttpRequest.newBuilder(URI.create(baseUrl + "/ready"))
                        .POST(HttpRequest.BodyPublishers.noBody()).build(),
                HttpResponse.BodyHandlers.ofString());

        assertEquals(405, response.statusCode());
    }

    @Test
    void fraudScoreReturnsContract() throws Exception {
        HttpResponse<String> response = postFraudScore(SAMPLE_PAYLOAD);

        assertEquals(200, response.statusCode());
        JsonNode body = MAPPER.readTree(response.body());
        assertTrue(body.has("approved"), "response must contain 'approved'");
        assertTrue(body.has("fraud_score"), "response must contain 'fraud_score'");
        assertTrue(body.get("approved").isBoolean());
        assertTrue(body.get("fraud_score").isNumber());
    }

    @Test
    void fraudScoreAcceptsNullLastTransaction() throws Exception {
        String payload = SAMPLE_PAYLOAD.replaceFirst(
                "(?s)\"last_transaction\": \\{.*?\\}", "\"last_transaction\": null");

        HttpResponse<String> response = postFraudScore(payload);

        assertEquals(200, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).has("fraud_score"));
    }

    @Test
    void fraudScoreRejectsMissingSection() throws Exception {
        HttpResponse<String> response = postFraudScore("{\"id\":\"tx-1\"}");

        assertEquals(400, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).has("error"));
    }

    @Test
    void fraudScoreRejectsInvalidJson() throws Exception {
        HttpResponse<String> response = postFraudScore("not json");

        assertEquals(400, response.statusCode());
        assertTrue(MAPPER.readTree(response.body()).has("error"));
    }

    @Test
    void fraudScoreRejectsGet() throws Exception {
        HttpResponse<String> response = client.send(
                HttpRequest.newBuilder(URI.create(baseUrl + "/fraud-score")).GET().build(),
                HttpResponse.BodyHandlers.ofString());

        assertEquals(405, response.statusCode());
    }

    private static HttpResponse<String> postFraudScore(String payload) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(baseUrl + "/fraud-score"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(payload))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }
}
