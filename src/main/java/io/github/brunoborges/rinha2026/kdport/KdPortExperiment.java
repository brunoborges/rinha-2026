package io.github.brunoborges.rinha2026.kdport;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonGenerator;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.brunoborges.rinha2026.ReferenceDataset;
import io.github.brunoborges.rinha2026.RequestVectorParser;
import io.github.brunoborges.rinha2026.TransactionVectorizer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Offline-only kdport validation harness; never wired into the server. */
public final class KdPortExperiment {
    private static final int DIMS = Dataset.DIMS;
    private static final int K = 5;

    public static void main(String[] args) throws Exception {
        Path bin = Path.of(args.length > 0 ? args[0] : "resources/references.bin");
        Path data = Path.of(args.length > 1 ? args[1] : "test/test-data.json");

        System.out.println("env: exact baseline uses no KDTREE_* relaxation/cap env vars");
        System.out.println("env: production example: KDTREE_PROFILING=1 KDTREE_EARLY_DIST_MILLI=110 "
                + "KDTREE_REFINE_BOUNDARY=1 KDTREE_PRIME_PLUNGE_CAP=1 KDTREE_RELAX_SOFT_CAP=700 "
                + "KDTREE_RELAX_RANGE=900 KDTREE_RELAX_EPSILON=0.025 KDTREE_BUCKET_LEAF_DEPTH=18 "
                + "KDTREE_BUCKET_LEAF_MAX_NODES=64");

        TransactionVectorizer vectorizer = new TransactionVectorizer();
        List<double[]> qvecs = new ArrayList<>();
        List<Boolean> expectedApproved = new ArrayList<>();
        loadCorpus(data, vectorizer, qvecs, expectedApproved);
        System.out.printf("corpus: %,d queries%n", qvecs.size());

        try (ReferenceDataset refs = ReferenceDataset.mmap(bin)) {
            refs.preload();
            System.out.printf("references: %,d vectors%n", refs.count());
            long buildStart = System.nanoTime();
            KdTree tree = KdTreeBuilder.build(Dataset.fromReferenceDataset(refs));
            long buildNanos = System.nanoTime() - buildStart;
            System.out.printf("build_ms=%d%n", buildNanos / 1_000_000);

            float[] query = new float[DIMS];
            long[] visits = new long[qvecs.size()];
            int fp = 0;
            int fn = 0;
            long queryStart = System.nanoTime();
            for (int i = 0; i < qvecs.size(); i++) {
                double[] src = qvecs.get(i);
                for (int d = 0; d < DIMS; d++) query[d] = (float) src[d];
                int frauds = tree.countFraudsInTop5Fast(query);
                boolean predictedFraud = frauds >= 3;
                boolean actualFraud = !expectedApproved.get(i);
                if (predictedFraud && !actualFraud) fp++;
                else if (!predictedFraud && actualFraud) fn++;
                visits[i] = KdTreeProbes.lastQueryNodesVisited();
            }
            long queryNanos = System.nanoTime() - queryStart;
            report(fp, fn, visits, buildNanos, queryNanos);
        }
    }

    private static void report(int fp, int fn, long[] visits, long buildNanos, long queryNanos) {
        long[] sorted = visits.clone();
        Arrays.sort(sorted);
        long sum = 0;
        for (long v : visits) sum += v;
        double mean = (double) sum / visits.length;
        double qps = visits.length / (queryNanos / 1_000_000_000.0);
        System.out.printf("FP=%d FN=%d E=%d%n", fp, fn, fp + 3 * fn);
        System.out.printf("visits p50=%d p95=%d p99=%d max=%d mean=%.2f%n",
                pct(sorted, 50), pct(sorted, 95), pct(sorted, 99), sorted[sorted.length - 1], mean);
        System.out.printf("build_ms=%d query_ms=%d qps=%.2f%n",
                buildNanos / 1_000_000, queryNanos / 1_000_000, qps);
    }

    private static long pct(long[] sorted, int p) {
        int idx = (int) Math.min(sorted.length - 1L, (long) Math.ceil(p / 100.0 * sorted.length) - 1);
        return sorted[Math.max(0, idx)];
    }

    static void loadCorpus(Path data, TransactionVectorizer vectorizer,
                           List<double[]> qvecs, List<Boolean> expected) throws Exception {
        JsonFactory factory = new JsonFactory();
        RequestVectorParser parser = new RequestVectorParser(vectorizer);
        try (JsonParser p = factory.createParser(Files.newInputStream(data))) {
            p.nextToken();
            while (p.nextToken() != JsonToken.END_OBJECT) {
                if (!"entries".equals(p.currentName())) { p.nextToken(); p.skipChildren(); continue; }
                p.nextToken();
                while (p.nextToken() != JsonToken.END_ARRAY) {
                    byte[] reqBytes = null; Boolean approved = null;
                    while (p.nextToken() != JsonToken.END_OBJECT) {
                        switch (p.currentName()) {
                            case "request" -> {
                                p.nextToken();
                                ByteArrayOutputStream buf = new ByteArrayOutputStream(512);
                                try (JsonGenerator g = factory.createGenerator(buf)) { g.copyCurrentStructure(p); }
                                reqBytes = buf.toByteArray();
                            }
                            case "expected_approved" -> approved = p.nextBooleanValue();
                            default -> { p.nextToken(); p.skipChildren(); }
                        }
                    }
                    RequestVectorParser.State st = parser.acquire();
                    try {
                        parser.vectorize(new ByteArrayInputStream(reqBytes), st);
                        qvecs.add(Arrays.copyOf(st.qvec, DIMS));
                        expected.add(approved);
                    } finally {
                        parser.release(st);
                    }
                }
            }
        }
    }

    private KdPortExperiment() {
    }
}
