package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Reflection-free reader that turns a {@code /fraud-score} request body into a
 * {@link FraudRequest} using the jackson-core streaming API.
 *
 * <p>Replaces Jackson {@code databind} on the production path so the service can
 * be compiled to a GraalVM Native Image without any reflection configuration.
 * Unknown fields are skipped; absent sections stay {@code null} (so
 * {@link App#validate} can reject incomplete payloads); a {@code null}
 * {@code last_transaction} is preserved. Malformed JSON throws {@link IOException},
 * which the server maps to {@code HTTP 400}.
 */
final class FraudRequestParser {

    private static final JsonFactory FACTORY = new JsonFactory();

    private FraudRequestParser() {
    }

    static FraudRequest parse(InputStream in) throws IOException {
        try (JsonParser p = FACTORY.createParser(in)) {
            if (p.nextToken() != JsonToken.START_OBJECT) {
                throw new IOException("expected a JSON object");
            }
            String id = null;
            FraudRequest.Transaction transaction = null;
            FraudRequest.Customer customer = null;
            FraudRequest.Merchant merchant = null;
            FraudRequest.Terminal terminal = null;
            FraudRequest.LastTransaction lastTransaction = null;

            while (p.nextToken() != JsonToken.END_OBJECT) {
                String field = p.currentName();
                JsonToken value = p.nextToken();
                switch (field) {
                    case "id" -> id = text(p, value);
                    case "transaction" -> transaction = readTransaction(p, value);
                    case "customer" -> customer = readCustomer(p, value);
                    case "merchant" -> merchant = readMerchant(p, value);
                    case "terminal" -> terminal = readTerminal(p, value);
                    case "last_transaction" -> lastTransaction = readLastTransaction(p, value);
                    default -> p.skipChildren();
                }
            }
            return new FraudRequest(id, transaction, customer, merchant, terminal, lastTransaction);
        }
    }

    private static FraudRequest.Transaction readTransaction(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        expectObject(value);
        double amount = 0;
        int installments = 0;
        Instant requestedAt = null;
        while (p.nextToken() != JsonToken.END_OBJECT) {
            String field = p.currentName();
            JsonToken v = p.nextToken();
            switch (field) {
                case "amount" -> amount = p.getValueAsDouble();
                case "installments" -> installments = p.getValueAsInt();
                case "requested_at" -> requestedAt = instant(p, v);
                default -> p.skipChildren();
            }
        }
        return new FraudRequest.Transaction(amount, installments, requestedAt);
    }

    private static FraudRequest.Customer readCustomer(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        expectObject(value);
        double avgAmount = 0;
        int txCount24h = 0;
        List<String> knownMerchants = null;
        while (p.nextToken() != JsonToken.END_OBJECT) {
            String field = p.currentName();
            JsonToken v = p.nextToken();
            switch (field) {
                case "avg_amount" -> avgAmount = p.getValueAsDouble();
                case "tx_count_24h" -> txCount24h = p.getValueAsInt();
                case "known_merchants" -> knownMerchants = readStringArray(p, v);
                default -> p.skipChildren();
            }
        }
        return new FraudRequest.Customer(avgAmount, txCount24h, knownMerchants);
    }

    private static FraudRequest.Merchant readMerchant(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        expectObject(value);
        String id = null;
        String mcc = null;
        double avgAmount = 0;
        while (p.nextToken() != JsonToken.END_OBJECT) {
            String field = p.currentName();
            JsonToken v = p.nextToken();
            switch (field) {
                case "id" -> id = text(p, v);
                case "mcc" -> mcc = text(p, v);
                case "avg_amount" -> avgAmount = p.getValueAsDouble();
                default -> p.skipChildren();
            }
        }
        return new FraudRequest.Merchant(id, mcc, avgAmount);
    }

    private static FraudRequest.Terminal readTerminal(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        expectObject(value);
        boolean isOnline = false;
        boolean cardPresent = false;
        double kmFromHome = 0;
        while (p.nextToken() != JsonToken.END_OBJECT) {
            String field = p.currentName();
            JsonToken v = p.nextToken();
            switch (field) {
                case "is_online" -> isOnline = p.getValueAsBoolean();
                case "card_present" -> cardPresent = p.getValueAsBoolean();
                case "km_from_home" -> kmFromHome = p.getValueAsDouble();
                default -> p.skipChildren();
            }
        }
        return new FraudRequest.Terminal(isOnline, cardPresent, kmFromHome);
    }

    private static FraudRequest.LastTransaction readLastTransaction(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        expectObject(value);
        Instant timestamp = null;
        double kmFromCurrent = 0;
        while (p.nextToken() != JsonToken.END_OBJECT) {
            String field = p.currentName();
            JsonToken v = p.nextToken();
            switch (field) {
                case "timestamp" -> timestamp = instant(p, v);
                case "km_from_current" -> kmFromCurrent = p.getValueAsDouble();
                default -> p.skipChildren();
            }
        }
        return new FraudRequest.LastTransaction(timestamp, kmFromCurrent);
    }

    private static List<String> readStringArray(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        if (value != JsonToken.START_ARRAY) {
            throw new IOException("expected a JSON array");
        }
        List<String> values = new ArrayList<>();
        while (p.nextToken() != JsonToken.END_ARRAY) {
            values.add(p.getText());
        }
        return values;
    }

    private static String text(JsonParser p, JsonToken value) throws IOException {
        return value == JsonToken.VALUE_NULL ? null : p.getText();
    }

    private static Instant instant(JsonParser p, JsonToken value) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return null;
        }
        try {
            return Instant.parse(p.getText());
        } catch (RuntimeException e) {
            throw new IOException("invalid timestamp: " + p.getText());
        }
    }

    private static void expectObject(JsonToken value) throws IOException {
        if (value != JsonToken.START_OBJECT) {
            throw new IOException("expected a JSON object");
        }
    }
}
