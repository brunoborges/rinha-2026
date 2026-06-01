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

    /**
     * Whether this scorer can score directly from a pre-computed feature vector
     * via {@link #scoreVector(double[])}, letting the server parse a request
     * straight into the vector and skip the {@link FraudRequest} record graph.
     * Scorers that opt in must also return their {@link #vectorizer()}. The
     * default is {@code false} so diagnostic scorers keep the record path.
     */
    default boolean supportsVectorInput() {
        return false;
    }

    /**
     * The vectorizer this scorer uses, so the server can share a single instance
     * between the streaming parser and the scan. Only meaningful when
     * {@link #supportsVectorInput()} is {@code true}.
     */
    default TransactionVectorizer vectorizer() {
        return null;
    }

    /**
     * Scores an already-vectorized transaction. Only supported when
     * {@link #supportsVectorInput()} is {@code true}.
     *
     * @param qvec the 14-dimension feature vector (length &ge;
     *             {@link TransactionVectorizer#DIMENSIONS})
     * @throws UnsupportedOperationException if this scorer does not support
     *                                       vector input
     */
    default FraudResponse scoreVector(double[] qvec) {
        throw new UnsupportedOperationException("scorer does not support vector input");
    }
}
