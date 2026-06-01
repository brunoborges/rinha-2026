package io.github.brunoborges.rinha2026;

/**
 * Diagnostic {@link FraudScorer} that performs the full per-request
 * vectorization and quantization (the 14-dimension feature extraction) but
 * <em>skips the IVF candidate scan</em>, returning a constant response.
 *
 * <p>It exists purely for bottleneck attribution: running the load test with
 * {@code SCORER=stub} (no vectorize, no scan), {@code SCORER=vectorize} (this
 * scorer: vectorize only) and the default {@code SCORER=ivf} (full search)
 * decomposes per-request CPU and tail latency into three stages &mdash;
 * HTTP/parse/serialize, feature extraction, and nearest-neighbor scan &mdash;
 * so we can see which one drives CPU throttling. It is never selected unless
 * the {@code SCORER} environment variable explicitly requests it.
 */
public final class VectorizeOnlyFraudScorer implements FraudScorer {

    /**
     * Blackhole sink: consuming a byte of the quantized vector prevents the
     * ahead-of-time compiler from dead-code-eliminating the vectorization we are
     * trying to measure.
     */
    public static volatile int sink;

    private final TransactionVectorizer vectorizer;

    public VectorizeOnlyFraudScorer(TransactionVectorizer vectorizer) {
        this.vectorizer = vectorizer;
    }

    @Override
    public FraudResponse score(FraudRequest request) {
        short[] q = ReferenceDataset.quantize(vectorizer.vectorize(request));
        sink ^= q[0] ^ q[q.length - 1];
        return new FraudResponse(true, 0.0);
    }
}
