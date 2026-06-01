package io.github.brunoborges.rinha2026;

import java.util.List;
import java.util.Map;

/**
 * Turns an incoming {@link FraudRequest} into the 14-dimension fraud detection
 * vector defined in {@code docs/DETECTION_RULES.md}.
 *
 * <p>The dimensions, in order, are:
 * <ol start="0">
 *   <li>{@code amount}</li>
 *   <li>{@code installments}</li>
 *   <li>{@code amount_vs_avg}</li>
 *   <li>{@code hour_of_day}</li>
 *   <li>{@code day_of_week}</li>
 *   <li>{@code minutes_since_last_tx} ({@code -1} when {@code last_transaction} is null)</li>
 *   <li>{@code km_from_last_tx} ({@code -1} when {@code last_transaction} is null)</li>
 *   <li>{@code km_from_home}</li>
 *   <li>{@code tx_count_24h}</li>
 *   <li>{@code is_online}</li>
 *   <li>{@code card_present}</li>
 *   <li>{@code unknown_merchant}</li>
 *   <li>{@code mcc_risk}</li>
 *   <li>{@code merchant_avg_amount}</li>
 * </ol>
 *
 * <p>Every dimension is normalized to {@code [0.0, 1.0]} via {@link #clamp(double)},
 * except indices 5 and 6 which use the sentinel {@code -1} to flag "no previous
 * transaction" (the only legal out-of-range value).
 */
public final class TransactionVectorizer {

    /** Number of dimensions produced for each transaction vector. */
    public static final int DIMENSIONS = 14;

    /** Sentinel used at indices 5 and 6 when {@code last_transaction} is null. */
    public static final double NO_HISTORY = -1.0;

    private static final long SECONDS_PER_MINUTE = 60L;
    private static final long SECONDS_PER_HOUR = 3_600L;
    private static final long SECONDS_PER_DAY = 86_400L;

    /**
     * Normalization constants from {@code resources/normalization.json}.
     */
    public record Constants(
            double maxAmount,
            double maxInstallments,
            double amountVsAvgRatio,
            double maxMinutes,
            double maxKm,
            double maxTxCount24h,
            double maxMerchantAvgAmount) {

        /** Defaults documented in {@code docs/DATASET.md}. */
        public static final Constants DEFAULTS = new Constants(
                10_000, 12, 10, 1_440, 1_000, 20, 10_000);
    }

    /** Risk applied to MCCs absent from {@link #mccRisk}. */
    public static final double DEFAULT_MCC_RISK = 0.5;

    /** Risk score by MCC from {@code resources/mcc_risk.json}. */
    public static final Map<String, Double> DEFAULT_MCC_RISK_MAP = Map.ofEntries(
            Map.entry("5411", 0.15),
            Map.entry("5812", 0.30),
            Map.entry("5912", 0.20),
            Map.entry("5944", 0.45),
            Map.entry("7801", 0.80),
            Map.entry("7802", 0.75),
            Map.entry("7995", 0.85),
            Map.entry("4511", 0.35),
            Map.entry("5311", 0.25),
            Map.entry("5999", 0.50));

    private final Constants constants;
    private final Map<String, Double> mccRisk;

    /** Creates a vectorizer with the documented defaults. */
    public TransactionVectorizer() {
        this(Constants.DEFAULTS, DEFAULT_MCC_RISK_MAP);
    }

    /**
     * Creates a vectorizer with custom normalization constants and MCC risk map.
     *
     * @param constants normalization constants (not {@code null})
     * @param mccRisk   risk score per MCC; missing keys fall back to
     *                  {@link #DEFAULT_MCC_RISK}
     */
    public TransactionVectorizer(Constants constants, Map<String, Double> mccRisk) {
        this.constants = constants;
        this.mccRisk = Map.copyOf(mccRisk);
    }

    /**
     * Vectorizes and normalizes a transaction into its 14-dimension vector.
     *
     * @param request the incoming transaction payload (not {@code null})
     * @return a freshly allocated array of length {@link #DIMENSIONS}
     */
    public double[] vectorize(FraudRequest request) {
        double[] v = new double[DIMENSIONS];
        vectorizeInto(request, v);
        return v;
    }

    /**
     * Allocation-free variant of {@link #vectorize(FraudRequest)}: writes the
     * 14-dimension vector into the caller-supplied {@code v} (length &ge;
     * {@link #DIMENSIONS}). Every dimension is written unconditionally so a
     * reused buffer never leaks values from a previous request.
     *
     * <p>Hour-of-day and day-of-week are derived from the UTC epoch second with
     * pure integer arithmetic rather than {@code requestedAt().atZone(UTC)}, and
     * the minutes-since-last-transaction from an epoch-second subtraction rather
     * than {@link java.time.Duration} &mdash; both avoid the {@code java.time}
     * object graph that would otherwise be allocated and discarded per request.
     *
     * @param request the incoming transaction payload (not {@code null})
     * @param v       the destination buffer (length &ge; {@link #DIMENSIONS})
     */
    public void vectorizeInto(FraudRequest request, double[] v) {
        FraudRequest.Transaction tx = request.transaction();
        FraudRequest.Customer customer = request.customer();
        FraudRequest.Merchant merchant = request.merchant();
        FraudRequest.Terminal terminal = request.terminal();
        FraudRequest.LastTransaction last = request.lastTransaction();
        boolean hasLast = last != null;

        vectorizeInto(
                tx.amount(), tx.installments(), tx.requestedAt().getEpochSecond(),
                customer.avgAmount(), customer.txCount24h(),
                isUnknownMerchant(merchant.id(), customer.knownMerchants()),
                merchant.mcc(), merchant.avgAmount(), terminal.kmFromHome(),
                terminal.isOnline(), terminal.cardPresent(),
                hasLast, hasLast ? last.timestamp().getEpochSecond() : 0L,
                hasLast ? last.kmFromCurrent() : 0.0,
                v);
    }

    /**
     * Primitive-core variant of {@link #vectorizeInto(FraudRequest, double[])}
     * that takes the already-extracted scalar features instead of the
     * {@link FraudRequest} record graph. The streaming hot path
     * ({@link RequestVectorParser}) calls this directly so a request can be parsed
     * straight into its feature vector without allocating the record graph, the
     * {@code known_merchants} list, or the {@link java.time.Instant} objects. The
     * {@link FraudRequest} overload delegates here, so both paths share identical
     * math.
     *
     * @param amount              transaction amount
     * @param installments        number of installments
     * @param requestedEpoch      transaction time as a UTC epoch second
     * @param customerAvgAmount   customer average transaction amount
     * @param txCount24h          customer transaction count in the last 24h
     * @param unknownMerchant     whether the merchant is outside the customer's
     *                            {@code known_merchants}
     * @param mcc                 merchant category code (may be {@code null})
     * @param merchantAvgAmount   merchant average transaction amount
     * @param kmFromHome          terminal distance from the customer's home
     * @param isOnline            whether the terminal is online
     * @param cardPresent         whether the card was present
     * @param hasLast             whether a previous transaction is known
     * @param lastEpoch           previous transaction time (UTC epoch second);
     *                            ignored when {@code hasLast} is {@code false}
     * @param lastKmFromCurrent   distance from the previous transaction; ignored
     *                            when {@code hasLast} is {@code false}
     * @param v                   the destination buffer (length &ge;
     *                            {@link #DIMENSIONS})
     */
    public void vectorizeInto(
            double amount, int installments, long requestedEpoch,
            double customerAvgAmount, int txCount24h, boolean unknownMerchant,
            String mcc, double merchantAvgAmount, double kmFromHome,
            boolean isOnline, boolean cardPresent,
            boolean hasLast, long lastEpoch, double lastKmFromCurrent,
            double[] v) {
        // UTC hour-of-day (00:00 == epoch second 0) and ISO day-of-week
        // (Mon=1..Sun=7; epoch day 0 == 1970-01-01 == Thursday == 4).
        int hour = (int) (Math.floorMod(requestedEpoch, SECONDS_PER_DAY) / SECONDS_PER_HOUR);
        int dayOfWeek = (int) Math.floorMod(Math.floorDiv(requestedEpoch, SECONDS_PER_DAY) + 3, 7) + 1;

        v[0] = clamp(amount / constants.maxAmount());
        v[1] = clamp(installments / constants.maxInstallments());
        v[2] = clamp((amount / customerAvgAmount) / constants.amountVsAvgRatio());
        v[3] = hour / 23.0;
        v[4] = (dayOfWeek - 1) / 6.0;

        if (!hasLast) {
            v[5] = NO_HISTORY;
            v[6] = NO_HISTORY;
        } else {
            long minutes = (requestedEpoch - lastEpoch) / SECONDS_PER_MINUTE;
            v[5] = clamp(minutes / constants.maxMinutes());
            v[6] = clamp(lastKmFromCurrent / constants.maxKm());
        }

        v[7] = clamp(kmFromHome / constants.maxKm());
        v[8] = clamp(txCount24h / constants.maxTxCount24h());
        v[9] = isOnline ? 1.0 : 0.0;
        v[10] = cardPresent ? 1.0 : 0.0;
        v[11] = unknownMerchant ? 1.0 : 0.0;
        v[12] = mccRisk.getOrDefault(mcc, DEFAULT_MCC_RISK);
        v[13] = clamp(merchantAvgAmount / constants.maxMerchantAvgAmount());
    }

    private static boolean isUnknownMerchant(String merchantId, List<String> knownMerchants) {
        return knownMerchants == null || !knownMerchants.contains(merchantId);
    }

    /**
     * Constrains a value to the {@code [0.0, 1.0]} interval.
     *
     * @param x the raw value
     * @return {@code 0.0} if below the range, {@code 1.0} if above, else {@code x}
     */
    public static double clamp(double x) {
        if (x < 0.0) {
            return 0.0;
        }
        return Math.min(x, 1.0);
    }
}
