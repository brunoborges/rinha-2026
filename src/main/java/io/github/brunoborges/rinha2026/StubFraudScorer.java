package io.github.brunoborges.rinha2026;

/**
 * Placeholder fraud scorer.
 *
 * <p>The real detection logic (vectorization + vector search, per the
 * specification) is not implemented yet. Until then this scorer returns a
 * neutral, deterministic response so the API contract is fully exercised.
 */
public final class StubFraudScorer implements FraudScorer {

    @Override
    public FraudResponse score(FraudRequest request) {
        // TBD: replace with the 14-dimension vectorization + vector search.
        return new FraudResponse(true, 0.0);
    }
}
