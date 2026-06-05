package io.github.brunoborges.rinha2026;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.charset.StandardCharsets;

/**
 * Pre-encoded, full HTTP/1.1 responses for the two contest routes.
 *
 * <p>{@code POST /fraud-score} yields a {@code fraud_score} of one of
 * {@code {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}} with {@code approved == fraud_score < 0.6}
 * (see {@link IvfFraudScorer#THRESHOLD}), so the six possible bodies are framed once at class load
 * together with the readiness / error responses. The hot path never builds a response string: it maps
 * a fraud-neighbour count (0..5) to a stable index and writes the pre-baked bytes.
 *
 * <p>Each response is materialised both as an on-heap {@code byte[]} (used by host tests and by the
 * record-path fallback) and as an off-heap {@link MemorySegment} (used by the Linux FFM
 * {@code send} transport, so the event loop writes straight from native memory with no per-response
 * copy or {@code ByteBuffer} allocation).
 */
final class HttpResponses {

    static final int RESP_FRAUD_0 = 0;
    static final int RESP_FRAUD_1 = 1;
    static final int RESP_FRAUD_2 = 2;
    static final int RESP_FRAUD_3 = 3;
    static final int RESP_FRAUD_4 = 4;
    static final int RESP_FRAUD_5 = 5;
    static final int RESP_READY = 6;
    static final int RESP_NOT_READY = 7;
    static final int RESP_NOT_FOUND = 8;
    static final int RESP_METHOD_NOT_ALLOWED = 9;
    static final int RESP_BAD_REQUEST = 10;
    static final int RESP_PAYLOAD_TOO_LARGE = 11;
    static final int RESP_COUNT = 12;

    /** Fail-safe response when scoring throws unexpectedly: deny (fraud_score 0.6) rather than surface a 5xx. */
    static final int RESP_FAIL_SAFE = RESP_FRAUD_3;

    private static final String[] FRAUD_SCORE = {"0.0", "0.2", "0.4", "0.6", "0.8", "1.0"};

    private static final byte[][] BYTES = new byte[RESP_COUNT][];
    private static final int[] LENGTHS = new int[RESP_COUNT];
    private static final MemorySegment[] SEGMENTS = new MemorySegment[RESP_COUNT];

    static {
        for (int frauds = 0; frauds <= 5; frauds++) {
            boolean approved = frauds < 3; // fraud_score < 0.6
            install(frauds, 200, "OK",
                    "{\"approved\":" + approved + ",\"fraud_score\":" + FRAUD_SCORE[frauds] + "}");
        }
        install(RESP_READY, 200, "OK", "{\"status\":\"ready\"}");
        install(RESP_NOT_READY, 503, "Service Unavailable", "{\"status\":\"starting\"}");
        install(RESP_NOT_FOUND, 404, "Not Found", "{\"error\":\"not found\"}");
        install(RESP_METHOD_NOT_ALLOWED, 405, "Method Not Allowed", "{\"error\":\"method not allowed\"}");
        install(RESP_BAD_REQUEST, 400, "Bad Request", "{\"error\":\"bad request\"}");
        install(RESP_PAYLOAD_TOO_LARGE, 413, "Payload Too Large", "{\"error\":\"payload too large\"}");
    }

    private static void install(int idx, int status, String reason, String body) {
        byte[] bodyBytes = body.getBytes(StandardCharsets.UTF_8);
        String head = "HTTP/1.1 " + status + " " + reason + "\r\n"
                + "Content-Type: application/json\r\n"
                + "Content-Length: " + bodyBytes.length + "\r\n"
                + "\r\n";
        byte[] headBytes = head.getBytes(StandardCharsets.UTF_8);
        byte[] full = new byte[headBytes.length + bodyBytes.length];
        System.arraycopy(headBytes, 0, full, 0, headBytes.length);
        System.arraycopy(bodyBytes, 0, full, headBytes.length, bodyBytes.length);

        BYTES[idx] = full;
        LENGTHS[idx] = full.length;
        // Global arena: these constants live for the whole process; no per-request native allocation.
        MemorySegment seg = Arena.global().allocate(full.length);
        MemorySegment.copy(full, 0, seg, ValueLayout.JAVA_BYTE, 0, full.length);
        SEGMENTS[idx] = seg;
    }

    /** Map a fraud-neighbour count (0..5) to its response index. {@code RESP_FRAUD_0..5} are {@code 0..5}. */
    static int fraudCountToIndex(int fraudCount) {
        return fraudCount;
    }

    static byte[] bytes(int idx) {
        return BYTES[idx];
    }

    static int length(int idx) {
        return LENGTHS[idx];
    }

    /** Off-heap copy of the response, for FFM {@code send}. Read-only by convention; never mutated. */
    static MemorySegment segment(int idx) {
        return SEGMENTS[idx];
    }

    private HttpResponses() {
    }
}
