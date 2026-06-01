package io.github.brunoborges.rinha2026;

/**
 * Strategy for scoring a transaction's fraud likelihood.
 */
@FunctionalInterface
public interface FraudScorer {

    FraudResponse score(FraudRequest request);

    /**
     * Optionally prepares the scorer for full-speed serving before the server
     * accepts traffic (e.g. faulting a memory-mapped dataset into the page
     * cache). The default is a no-op.
     */
    default void preload() {
    }
}
