package io.github.brunoborges.rinha2026;

import java.io.ByteArrayInputStream;
import java.io.IOException;

/**
 * Maps a parsed {@link HttpConnection} to a pre-framed {@link HttpResponses} index, bridging the hot
 * {@code POST /fraud-score} path to the {@link FraudScorer}.
 *
 * <p>Single-threaded by contract: the Linux event loop is the only caller, so the vector path reuses a
 * single {@link RequestVectorParser.State} (no pool, no per-request allocation). Scorers that cannot take a
 * pre-computed vector (the {@code stub}/{@code vectorize} diagnostics) fall back to the
 * {@link FraudRequestParser} record path, which is off the contest hot path.
 *
 * <p>Readiness is gated by {@link #markReady()} so {@code GET /ready} only returns 200 once the dataset
 * has been faulted into the page cache and the server is actually serving.
 */
final class HttpRouter {

    private final FraudScorer scorer;
    private final RequestVectorParser parser;
    private final RequestVectorParser.State state;

    private volatile boolean ready;

    HttpRouter(FraudScorer scorer) {
        this.scorer = scorer;
        if (scorer.supportsVectorInput()) {
            this.parser = new RequestVectorParser(scorer.vectorizer());
            this.state = parser.newState();
        } else {
            this.parser = null;
            this.state = null;
        }
    }

    void markReady() {
        ready = true;
    }

    /** Resolve the response index for a fully-parsed request. */
    int responseIndex(HttpConnection c) {
        return switch (c.routeId) {
            case HttpConnection.ROUTE_FRAUD_SCORE -> scoreFraud(c.buf, c.bodyStart, c.bodyLen);
            case HttpConnection.ROUTE_READY -> ready ? HttpResponses.RESP_READY : HttpResponses.RESP_NOT_READY;
            case HttpConnection.ROUTE_METHOD_NOT_ALLOWED -> HttpResponses.RESP_METHOD_NOT_ALLOWED;
            default -> HttpResponses.RESP_NOT_FOUND;
        };
    }

    private int scoreFraud(byte[] body, int off, int len) {
        if (parser != null) {
            try {
                parser.vectorize(body, off, len, state);
                FraudResponse r = scorer.scoreVector(state.qvec);
                return scoreToIndex(r);
            } catch (IOException badRequest) {
                return HttpResponses.RESP_BAD_REQUEST;
            } catch (Throwable unexpected) {
                return HttpResponses.RESP_FAIL_SAFE;
            }
        }
        return scoreFromRecord(body, off, len);
    }

    /** Record-path fallback for diagnostic scorers (stub/vectorize); not used in production scoring. */
    private int scoreFromRecord(byte[] body, int off, int len) {
        try {
            FraudRequest request = FraudRequestParser.parse(new ByteArrayInputStream(body, off, len));
            if (validate(request) != null) {
                return HttpResponses.RESP_BAD_REQUEST;
            }
            return scoreToIndex(scorer.score(request));
        } catch (IOException badRequest) {
            return HttpResponses.RESP_BAD_REQUEST;
        } catch (Throwable unexpected) {
            return HttpResponses.RESP_FAIL_SAFE;
        }
    }

    /** {@code fraud_score} is always {@code frauds/5}; recover the 0..5 index (clamped for safety). */
    private static int scoreToIndex(FraudResponse r) {
        int idx = (int) Math.round(r.fraudScore() * 5.0);
        if (idx < 0) {
            return 0;
        }
        return Math.min(idx, 5);
    }

    private static String validate(FraudRequest request) {
        if (request == null) {
            return "request body must be a JSON object";
        }
        if (request.id() == null) {
            return "missing required field: id";
        }
        if (request.transaction() == null) {
            return "missing required field: transaction";
        }
        if (request.customer() == null) {
            return "missing required field: customer";
        }
        if (request.merchant() == null) {
            return "missing required field: merchant";
        }
        if (request.terminal() == null) {
            return "missing required field: terminal";
        }
        return null;
    }
}
