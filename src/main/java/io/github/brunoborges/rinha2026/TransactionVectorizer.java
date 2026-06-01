package io.github.brunoborges.rinha2026;

import java.time.ZoneOffset;
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
        FraudRequest.Transaction tx = request.transaction();
        FraudRequest.Customer customer = request.customer();
        FraudRequest.Merchant merchant = request.merchant();
        FraudRequest.Terminal terminal = request.terminal();
        FraudRequest.LastTransaction last = request.lastTransaction();

        var dateTime = tx.requestedAt().atZone(ZoneOffset.UTC);

        double[] v = new double[DIMENSIONS];

        v[0] = clamp(tx.amount() / constants.maxAmount());
        v[1] = clamp(tx.installments() / constants.maxInstallments());
        v[2] = clamp((tx.amount() / customer.avgAmount()) / constants.amountVsAvgRatio());
        v[3] = dateTime.getHour() / 23.0;
        v[4] = (dateTime.getDayOfWeek().getValue() - 1) / 6.0;

        if (last == null) {
            v[5] = NO_HISTORY;
            v[6] = NO_HISTORY;
        } else {
            double minutes = java.time.Duration.between(last.timestamp(), tx.requestedAt()).toMinutes();
            v[5] = clamp(minutes / constants.maxMinutes());
            v[6] = clamp(last.kmFromCurrent() / constants.maxKm());
        }

        v[7] = clamp(terminal.kmFromHome() / constants.maxKm());
        v[8] = clamp(customer.txCount24h() / constants.maxTxCount24h());
        v[9] = terminal.isOnline() ? 1.0 : 0.0;
        v[10] = terminal.cardPresent() ? 1.0 : 0.0;
        v[11] = isUnknownMerchant(merchant.id(), customer.knownMerchants()) ? 1.0 : 0.0;
        v[12] = mccRisk.getOrDefault(merchant.mcc(), DEFAULT_MCC_RISK);
        v[13] = clamp(merchant.avgAmount() / constants.maxMerchantAvgAmount());

        return v;
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
