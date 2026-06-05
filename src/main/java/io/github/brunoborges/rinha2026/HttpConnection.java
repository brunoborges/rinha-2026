package io.github.brunoborges.rinha2026;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Per-connection state and a single-pass, allocation-free HTTP/1.1 request parser. Recycled across
 * pipelined keep-alive requests on the same socket.
 *
 * <p>Transport-agnostic: the parser only sees a heap {@code byte[]} ({@link #buf}) filled to
 * {@link #pos} bytes. The Linux FFM event loop copies bytes off the socket into {@link #buf}; host
 * tests append bytes directly. This keeps all wire-format logic free of any OS dependency so it can
 * be unit-tested off-Linux even though the production transport is Linux-only.
 */
final class HttpConnection {

    static final int NEED_MORE = 0;
    static final int READY = 1;
    static final int MALFORMED = 2;
    static final int TOO_LARGE = 3;

    static final int ROUTE_NONE = 0;
    static final int ROUTE_FRAUD_SCORE = 1; // POST /fraud-score
    static final int ROUTE_READY = 2;       // GET /ready
    static final int ROUTE_METHOD_NOT_ALLOWED = 3; // known path, wrong method
    static final int ROUTE_NOT_FOUND = 4;   // unknown path

    static final int BUF_SIZE = 8192;

    private static final byte[] PATH_FRAUD = "/fraud-score".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] PATH_READY = "/ready".getBytes(StandardCharsets.US_ASCII);
    private static final byte[] CONTENT_LENGTH_LC = "content-length:".getBytes(StandardCharsets.US_ASCII);

    // SWAR scan for '\r' (0x0d): the JIT/AOT folds the unaligned 8-byte read; the hasZero trick locates any
    // matching byte across 8 lanes at once.
    private static final VarHandle LONG_LE =
            MethodHandles.byteArrayViewVarHandle(long[].class, ByteOrder.LITTLE_ENDIAN);
    private static final long CR_BROADCAST = 0x0d0d0d0d0d0d0d0dL;
    private static final long LOW_BITS = 0x0101010101010101L;
    private static final long HIGH_BITS = 0x8080808080808080L;

    final byte[] buf = new byte[BUF_SIZE];
    int pos = 0;

    int routeId = ROUTE_NONE;
    int bodyStart = -1;
    int bodyLen = -1;
    boolean closeAfterWrite = false;

    /**
     * Parse the buffered bytes. Idempotent once {@link #bodyStart} is set, so the caller may re-enter on the
     * next read when the body has not fully arrived.
     */
    int tryParse() {
        if (bodyStart < 0) {
            int n = pos;
            if (n < 4) {
                return NEED_MORE;
            }

            int cr0 = findCrLf(buf, 0, n);
            if (cr0 < 0) {
                return NEED_MORE;
            }
            routeId = parseRequestLine(buf, n, cr0);

            int contentLen = 0;
            int i = cr0 + 2;
            while (true) {
                if (i + 1 >= n) {
                    return NEED_MORE;
                }
                if (buf[i] == '\r' && buf[i + 1] == '\n') {
                    i += 2;
                    break;
                }
                if ((buf[i] | 0x20) == 'c' && hasContentLengthHeader(buf, i, n)) {
                    int v = i + CONTENT_LENGTH_LC.length;
                    while (v < n && (buf[v] == ' ' || buf[v] == '\t')) {
                        v++;
                    }
                    while (v < n && buf[v] >= '0' && buf[v] <= '9') {
                        contentLen = contentLen * 10 + (buf[v] - '0');
                        v++;
                    }
                }
                int next = findCrLf(buf, i, n);
                if (next < 0) {
                    return NEED_MORE;
                }
                i = next + 2;
            }

            bodyStart = i;
            bodyLen = contentLen;
            if (bodyStart + bodyLen > BUF_SIZE) {
                return TOO_LARGE;
            }
        }
        int have = pos - bodyStart;
        if (have < bodyLen) {
            return NEED_MORE;
        }
        return READY;
    }

    /**
     * Classify {@code METHOD SP PATH SP HTTP/x} within {@code [0, lineEnd)} into a route id. Only the two
     * contest routes are matched exactly; everything else maps to 404, and a known path with the wrong method
     * maps to 405.
     */
    private static int parseRequestLine(byte[] data, int n, int lineEnd) {
        int sp1 = indexOf(data, 0, lineEnd, (byte) ' ');
        if (sp1 < 0) {
            return ROUTE_NOT_FOUND;
        }
        int pathStart = sp1 + 1;
        int sp2 = indexOf(data, pathStart, lineEnd, (byte) ' ');
        int pathEnd = sp2 < 0 ? lineEnd : sp2;

        if (regionEquals(data, pathStart, pathEnd, PATH_FRAUD)) {
            return isMethod(data, 0, sp1, 'P', 'O', 'S', 'T') ? ROUTE_FRAUD_SCORE : ROUTE_METHOD_NOT_ALLOWED;
        }
        if (regionEquals(data, pathStart, pathEnd, PATH_READY)) {
            return isMethodGet(data, 0, sp1) ? ROUTE_READY : ROUTE_METHOD_NOT_ALLOWED;
        }
        return ROUTE_NOT_FOUND;
    }

    private static boolean isMethod(byte[] d, int from, int to, char a, char b, char c, char e) {
        return to - from == 4 && d[from] == a && d[from + 1] == b && d[from + 2] == c && d[from + 3] == e;
    }

    private static boolean isMethodGet(byte[] d, int from, int to) {
        return to - from == 3 && d[from] == 'G' && d[from + 1] == 'E' && d[from + 2] == 'T';
    }

    private static boolean regionEquals(byte[] d, int from, int to, byte[] lit) {
        if (to - from != lit.length) {
            return false;
        }
        for (int i = 0; i < lit.length; i++) {
            if (d[from + i] != lit[i]) {
                return false;
            }
        }
        return true;
    }

    private static int indexOf(byte[] d, int from, int to, byte target) {
        for (int i = from; i < to; i++) {
            if (d[i] == target) {
                return i;
            }
        }
        return -1;
    }

    /** Offset of '\r' such that {@code data[cr+1] == '\n'} within {@code [from, limit)}, or -1. */
    private static int findCrLf(byte[] data, int from, int limit) {
        int i = from;
        while (i < limit) {
            int cr = indexOfCr(data, i, limit);
            if (cr < 0 || cr + 1 >= limit) {
                return -1;
            }
            if (data[cr + 1] == '\n') {
                return cr;
            }
            i = cr + 1;
        }
        return -1;
    }

    private static int indexOfCr(byte[] data, int from, int limit) {
        int i = from;
        int end = limit - 7;
        while (i <= end) {
            long w = (long) LONG_LE.get(data, i);
            long x = w ^ CR_BROADCAST;
            long m = (x - LOW_BITS) & ~x & HIGH_BITS;
            if (m != 0) {
                return i + (Long.numberOfTrailingZeros(m) >>> 3);
            }
            i += 8;
        }
        while (i < limit) {
            if (data[i] == '\r') {
                return i;
            }
            i++;
        }
        return -1;
    }

    private static boolean hasContentLengthHeader(byte[] data, int from, int n) {
        if (n - from < CONTENT_LENGTH_LC.length) {
            return false;
        }
        for (int i = 0; i < CONTENT_LENGTH_LC.length; i++) {
            byte expected = CONTENT_LENGTH_LC[i];
            byte actual = data[from + i];
            // Case-fold letters; ':' and '-' are unaffected by | 0x20.
            if (expected >= 'a' && expected <= 'z') {
                if ((actual | 0x20) != expected) {
                    return false;
                }
            } else if (actual != expected) {
                return false;
            }
        }
        return true;
    }

    /** Compact the buffer after a fully-served request, preserving any pipelined bytes of the next request. */
    void advanceAfterRequest() {
        int requestEnd = bodyStart + bodyLen;
        int extra = pos - requestEnd;
        if (extra > 0) {
            System.arraycopy(buf, requestEnd, buf, 0, extra);
        }
        pos = Math.max(extra, 0);
        routeId = ROUTE_NONE;
        bodyStart = -1;
        bodyLen = -1;
    }

    void reset() {
        pos = 0;
        routeId = ROUTE_NONE;
        bodyStart = -1;
        bodyLen = -1;
        closeAfterWrite = false;
    }
}
