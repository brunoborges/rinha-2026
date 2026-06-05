package io.github.brunoborges.rinha2026;

import java.nio.charset.StandardCharsets;
import java.util.Random;
import java.util.logging.Logger;

/**
 * Ahead-of-time (AOT) training harness for the JDK 25 AOT cache
 * ({@code -XX:AOTMode=record}). When {@code AOT_TRAINING=1}, {@link App#main}
 * runs this instead of starting the fd-passing server: it loads the real
 * reference-backed scorer and pushes a large batch of synthetic, schema-valid
 * {@code POST /fraud-score} requests through the <em>exact</em> in-process hot
 * path the server uses — {@link HttpConnection} parse → {@link HttpRouter}
 * routing → {@link FraudScorer#scoreVector} IVF scan → response selection — then
 * exits. This makes the JVM load/link the production classes and record method
 * profiles (JEP 515) for the genuinely hot methods (jackson streaming parse,
 * {@link TransactionVectorizer}, the int16 IVF scan), so the create step bakes
 * them into the AOT cache.
 *
 * <p>The synthetic socket-free path is sufficient because the CPU bottleneck is
 * the IVF distance scan, not socket I/O. Payload shapes vary (amounts,
 * installments, MCCs, online/card-present flags, distances, and a fraction with
 * no {@code last_transaction} to drive the {@code -1} sentinel branch) so the
 * profile covers both history and no-history code paths.
 */
final class TrainingDriver {

    private static final Logger LOG = Logger.getLogger(TrainingDriver.class.getName());

    /** Default number of synthetic requests to push through the pipeline. */
    private static final int DEFAULT_ITERATIONS = 300_000;

    private TrainingDriver() {
    }

    static void run() {
        int iterations = envInt("AOT_TRAINING_ITERS", DEFAULT_ITERATIONS);
        LOG.info(() -> "AOT training: driving " + iterations + " synthetic /fraud-score requests "
                + "through the in-process hot path");

        FraudScorer scorer = App.loadDefaultScorer();
        scorer.preload();
        HttpRouter router = new HttpRouter(scorer);
        router.markReady();

        HttpConnection conn = new HttpConnection();
        Random rnd = new Random(42);
        byte[] scratch = new byte[HttpConnection.BUF_SIZE];

        long start = System.nanoTime();
        int parsed = 0;
        for (int i = 0; i < iterations; i++) {
            int len = buildRequest(scratch, rnd);
            conn.reset();
            System.arraycopy(scratch, 0, conn.buf, 0, len);
            conn.pos = len;
            if (conn.tryParse() == HttpConnection.READY) {
                router.responseIndex(conn);
                parsed++;
            }
        }
        long ms = (System.nanoTime() - start) / 1_000_000;
        LOG.info("AOT training complete: " + parsed + " requests scored in " + ms + " ms; exiting");
        System.out.flush();
        System.exit(0);
    }

    /**
     * Writes a schema-valid {@code POST /fraud-score} HTTP/1.1 request into
     * {@code out} and returns its length. About one in eight requests omits
     * {@code last_transaction} to exercise the no-history ({@code -1} sentinel)
     * vectorizer branch.
     */
    private static int buildRequest(byte[] out, Random rnd) {
        boolean hasHistory = rnd.nextInt(8) != 0;
        StringBuilder b = new StringBuilder(512);
        b.append("{\"id\":\"tx-").append(rnd.nextInt(1_000_000_000)).append("\",");
        b.append("\"transaction\":{\"amount\":").append(money(rnd, 10_000))
                .append(",\"installments\":").append(1 + rnd.nextInt(12))
                .append(",\"requested_at\":\"2026-03-11T20:23:35Z\"},");
        b.append("\"customer\":{\"avg_amount\":").append(money(rnd, 5_000))
                .append(",\"tx_count_24h\":").append(rnd.nextInt(40))
                .append(",\"known_merchants\":[\"MERC-").append(pad3(rnd.nextInt(20)))
                .append("\",\"MERC-").append(pad3(rnd.nextInt(20))).append("\"]},");
        b.append("\"merchant\":{\"id\":\"MERC-").append(pad3(rnd.nextInt(20)))
                .append("\",\"mcc\":\"").append(MCCS[rnd.nextInt(MCCS.length)])
                .append("\",\"avg_amount\":").append(money(rnd, 3_000)).append("},");
        b.append("\"terminal\":{\"is_online\":").append(rnd.nextBoolean())
                .append(",\"card_present\":").append(rnd.nextBoolean())
                .append(",\"km_from_home\":").append(money(rnd, 5_000)).append("}");
        if (hasHistory) {
            b.append(",\"last_transaction\":{\"timestamp\":\"2026-03-11T14:58:35Z\",")
                    .append("\"km_from_current\":").append(money(rnd, 2_000)).append("}");
        }
        b.append("}");

        byte[] body = b.toString().getBytes(StandardCharsets.UTF_8);
        StringBuilder head = new StringBuilder(96);
        head.append("POST /fraud-score HTTP/1.1\r\nHost: t\r\n")
                .append("Content-Type: application/json\r\nContent-Length: ")
                .append(body.length).append("\r\n\r\n");
        byte[] headBytes = head.toString().getBytes(StandardCharsets.US_ASCII);

        int len = headBytes.length + body.length;
        if (len > out.length) {
            // Should not happen given the bounded fields; skip oversize by truncating safely.
            len = out.length;
        }
        System.arraycopy(headBytes, 0, out, 0, headBytes.length);
        System.arraycopy(body, 0, out, headBytes.length, Math.min(body.length, out.length - headBytes.length));
        return len;
    }

    private static final String[] MCCS = {"5912", "5411", "5812", "5999", "4111", "6011", "5732", "7995"};

    private static String money(Random rnd, int maxWhole) {
        return rnd.nextInt(maxWhole) + "." + pad2(rnd.nextInt(100));
    }

    private static String pad2(int v) {
        return v < 10 ? "0" + v : Integer.toString(v);
    }

    private static String pad3(int v) {
        return String.format("%03d", v);
    }

    private static int envInt(String name, int dflt) {
        String v = System.getenv(name);
        if (v == null) {
            return dflt;
        }
        try {
            return Integer.parseInt(v.trim());
        } catch (NumberFormatException e) {
            return dflt;
        }
    }
}
