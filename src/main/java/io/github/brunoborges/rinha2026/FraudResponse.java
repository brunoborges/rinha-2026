package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Response payload for {@code POST /fraud-score}.
 *
 * @param approved   whether the transaction is approved
 * @param fraudScore fraud likelihood in the range {@code [0.0, 1.0]}
 */
public record FraudResponse(
        @JsonProperty("approved") boolean approved,
        @JsonProperty("fraud_score") double fraudScore) {
}
