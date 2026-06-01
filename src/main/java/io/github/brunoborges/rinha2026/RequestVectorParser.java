package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.core.util.JsonRecyclerPools;

import java.io.IOException;
import java.io.InputStream;
import java.time.Instant;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;

/**
 * Allocation-light reader for the hot {@code /fraud-score} path: parses a request
 * body straight into a pooled 14-dimension feature vector using the jackson-core
 * streaming API, skipping the {@link FraudRequest} record graph, the
 * {@code known_merchants} list, and the {@link Instant} objects that
 * {@link FraudRequestParser} would otherwise allocate and immediately discard.
 *
 * <p>Validation is folded into the parse: a missing required section (or a
 * missing/invalid {@code transaction.requested_at}) throws {@link IOException},
 * which the server maps to {@code HTTP 400} &mdash; matching the rejection
 * behaviour of {@link FraudRequestParser} + {@link App#validate}. Because JSON
 * field order is not guaranteed, {@code known_merchants} is buffered into the
 * reused {@link State#km} array and the unknown-merchant test runs after the
 * object closes.
 *
 * <p>Working memory is recycled through a bounded {@link State} pool sized by the
 * {@code WORKERS} knob (see {@link Concurrency}), so after warmup the streaming
 * path allocates only the
 * short {@link String}s for the merchant id/mcc and {@code known_merchants}
 * entries (everything else is written into reused buffers). The jackson-core
 * {@link JsonFactory} is configured with a virtual-thread-friendly recycler pool
 * since the default {@code ThreadLocal} buffer recycling is ineffective with one
 * virtual thread per request.
 */
final class RequestVectorParser {

    private final JsonFactory factory;
    private final TransactionVectorizer vectorizer;
    private final ArrayBlockingQueue<State> pool;

    RequestVectorParser(TransactionVectorizer vectorizer) {
        this.vectorizer = vectorizer;
        this.factory = JsonFactory.builder()
                .recyclerPool(JsonRecyclerPools.newConcurrentDequePool())
                .build();
        int permits = Concurrency.workers();
        this.pool = new ArrayBlockingQueue<>(permits);
        for (int i = 0; i < permits; i++) {
            pool.add(new State());
        }
    }

    /** Borrows a {@link State} from the pool, retrying through interrupts. */
    State acquire() {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    return pool.take();
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    void release(State s) {
        pool.offer(s);
    }

    /**
     * Parses {@code in} straight into {@code s.qvec}. Throws {@link IOException}
     * (mapped to {@code HTTP 400}) for malformed JSON, a non-object root, a wrong
     * section shape, a missing required section, or a missing/invalid timestamp.
     */
    void vectorize(InputStream in, State s) throws IOException {
        boolean hasId = false;
        boolean hasTx = false;
        boolean hasCustomer = false;
        boolean hasMerchant = false;
        boolean hasTerminal = false;
        boolean hasRequestedAt = false;

        double amount = 0;
        int installments = 0;
        long requestedEpoch = 0;
        double custAvg = 0;
        int txCount24h = 0;
        s.kmCount = -1; // -1 == known_merchants absent/null -> unknown merchant
        String merchantId = null;
        String mcc = null;
        double merchantAvg = 0;
        boolean isOnline = false;
        boolean cardPresent = false;
        double kmFromHome = 0;
        boolean hasLast = false;
        long lastEpoch = 0;
        double lastKm = 0;

        try (JsonParser p = factory.createParser(in)) {
            if (p.nextToken() != JsonToken.START_OBJECT) {
                throw new IOException("request body must be a JSON object");
            }
            while (p.nextToken() != JsonToken.END_OBJECT) {
                String field = p.currentName();
                JsonToken v = p.nextToken();
                switch (field) {
                    case "id" -> {
                        if (v == JsonToken.START_OBJECT || v == JsonToken.START_ARRAY) {
                            p.skipChildren();
                        } else if (v != JsonToken.VALUE_NULL) {
                            hasId = true;
                        }
                    }
                    case "transaction" -> {
                        if (v == JsonToken.VALUE_NULL) {
                            continue;
                        }
                        expectObject(v);
                        hasTx = true;
                        while (p.nextToken() != JsonToken.END_OBJECT) {
                            String f = p.currentName();
                            JsonToken tv = p.nextToken();
                            switch (f) {
                                case "amount" -> amount = p.getValueAsDouble();
                                case "installments" -> installments = p.getValueAsInt();
                                case "requested_at" -> {
                                    if (tv != JsonToken.VALUE_NULL) {
                                        requestedEpoch = epochSeconds(p);
                                        hasRequestedAt = true;
                                    }
                                }
                                default -> p.skipChildren();
                            }
                        }
                    }
                    case "customer" -> {
                        if (v == JsonToken.VALUE_NULL) {
                            continue;
                        }
                        expectObject(v);
                        hasCustomer = true;
                        while (p.nextToken() != JsonToken.END_OBJECT) {
                            String f = p.currentName();
                            JsonToken cv = p.nextToken();
                            switch (f) {
                                case "avg_amount" -> custAvg = p.getValueAsDouble();
                                case "tx_count_24h" -> txCount24h = p.getValueAsInt();
                                case "known_merchants" -> readKnownMerchants(p, cv, s);
                                default -> p.skipChildren();
                            }
                        }
                    }
                    case "merchant" -> {
                        if (v == JsonToken.VALUE_NULL) {
                            continue;
                        }
                        expectObject(v);
                        hasMerchant = true;
                        while (p.nextToken() != JsonToken.END_OBJECT) {
                            String f = p.currentName();
                            JsonToken mv = p.nextToken();
                            switch (f) {
                                case "id" -> merchantId = (mv == JsonToken.VALUE_NULL) ? null : p.getText();
                                case "mcc" -> mcc = (mv == JsonToken.VALUE_NULL) ? null : p.getText();
                                case "avg_amount" -> merchantAvg = p.getValueAsDouble();
                                default -> p.skipChildren();
                            }
                        }
                    }
                    case "terminal" -> {
                        if (v == JsonToken.VALUE_NULL) {
                            continue;
                        }
                        expectObject(v);
                        hasTerminal = true;
                        while (p.nextToken() != JsonToken.END_OBJECT) {
                            String f = p.currentName();
                            p.nextToken();
                            switch (f) {
                                case "is_online" -> isOnline = p.getValueAsBoolean();
                                case "card_present" -> cardPresent = p.getValueAsBoolean();
                                case "km_from_home" -> kmFromHome = p.getValueAsDouble();
                                default -> p.skipChildren();
                            }
                        }
                    }
                    case "last_transaction" -> {
                        if (v == JsonToken.VALUE_NULL) {
                            continue;
                        }
                        expectObject(v);
                        hasLast = true;
                        while (p.nextToken() != JsonToken.END_OBJECT) {
                            String f = p.currentName();
                            JsonToken lv = p.nextToken();
                            switch (f) {
                                case "timestamp" -> {
                                    if (lv != JsonToken.VALUE_NULL) {
                                        lastEpoch = epochSeconds(p);
                                    }
                                }
                                case "km_from_current" -> lastKm = p.getValueAsDouble();
                                default -> p.skipChildren();
                            }
                        }
                    }
                    default -> p.skipChildren();
                }
            }
        }

        if (!hasId) {
            throw new IOException("missing required field: id");
        }
        if (!hasTx) {
            throw new IOException("missing required field: transaction");
        }
        if (!hasCustomer) {
            throw new IOException("missing required field: customer");
        }
        if (!hasMerchant) {
            throw new IOException("missing required field: merchant");
        }
        if (!hasTerminal) {
            throw new IOException("missing required field: terminal");
        }
        if (!hasRequestedAt) {
            throw new IOException("missing required field: transaction.requested_at");
        }

        boolean unknownMerchant = true;
        if (s.kmCount >= 0 && merchantId != null) {
            for (int i = 0; i < s.kmCount; i++) {
                if (merchantId.equals(s.km[i])) {
                    unknownMerchant = false;
                    break;
                }
            }
        }

        vectorizer.vectorizeInto(
                amount, installments, requestedEpoch, custAvg, txCount24h,
                unknownMerchant, mcc, merchantAvg, kmFromHome, isOnline, cardPresent,
                hasLast, lastEpoch, lastKm, s.qvec);
    }

    private static void readKnownMerchants(JsonParser p, JsonToken value, State s) throws IOException {
        if (value == JsonToken.VALUE_NULL) {
            return; // leave kmCount == -1 (treated as unknown merchant)
        }
        if (value != JsonToken.START_ARRAY) {
            throw new IOException("expected a JSON array");
        }
        int n = 0;
        while (p.nextToken() != JsonToken.END_ARRAY) {
            if (n == s.km.length) {
                s.km = Arrays.copyOf(s.km, s.km.length * 2);
            }
            s.km[n++] = p.getText();
        }
        s.kmCount = n;
    }

    private static long epochSeconds(JsonParser p) throws IOException {
        String text = p.getText();
        try {
            return Instant.parse(text).getEpochSecond();
        } catch (RuntimeException e) {
            throw new IOException("invalid timestamp: " + text);
        }
    }

    private static void expectObject(JsonToken value) throws IOException {
        if (value != JsonToken.START_OBJECT) {
            throw new IOException("expected a JSON object");
        }
    }

    /**
     * Per-request working memory recycled through the pool. {@code qvec} is the
     * destination feature vector; {@code km} buffers {@code known_merchants}
     * strings ({@code kmCount} entries, or {@code -1} when the field is absent).
     * Both are fully overwritten each request, so no state leaks between requests.
     */
    static final class State {
        final double[] qvec = new double[TransactionVectorizer.DIMENSIONS];
        String[] km = new String[16];
        int kmCount = -1;
    }
}
