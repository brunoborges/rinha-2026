package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.List;

/**
 * Request payload for {@code POST /fraud-score}.
 *
 * <p>Mirrors the contract defined in the Rinha de Backend 2026 API
 * specification. {@code last_transaction} may be {@code null}.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public record FraudRequest(
        @JsonProperty("id") String id,
        @JsonProperty("transaction") Transaction transaction,
        @JsonProperty("customer") Customer customer,
        @JsonProperty("merchant") Merchant merchant,
        @JsonProperty("terminal") Terminal terminal,
        @JsonProperty("last_transaction") LastTransaction lastTransaction) {

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Transaction(
            @JsonProperty("amount") double amount,
            @JsonProperty("installments") int installments,
            @JsonProperty("requested_at") Instant requestedAt) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Customer(
            @JsonProperty("avg_amount") double avgAmount,
            @JsonProperty("tx_count_24h") int txCount24h,
            @JsonProperty("known_merchants") List<String> knownMerchants) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Merchant(
            @JsonProperty("id") String id,
            @JsonProperty("mcc") String mcc,
            @JsonProperty("avg_amount") double avgAmount) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record Terminal(
            @JsonProperty("is_online") boolean isOnline,
            @JsonProperty("card_present") boolean cardPresent,
            @JsonProperty("km_from_home") double kmFromHome) {
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    public record LastTransaction(
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("km_from_current") double kmFromCurrent) {
    }
}
