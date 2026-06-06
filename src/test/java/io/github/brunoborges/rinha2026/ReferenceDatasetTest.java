package io.github.brunoborges.rinha2026;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReferenceDatasetTest {

    @Test
    void directReadDistanceMatchesPerElementOnMmap() throws Exception {
        ReferenceDataset source;
        try (InputStream in = getClass().getResourceAsStream("/example-references.json")) {
            source = ReferenceDataset.load(in);
        }
        Path bin = Files.createTempFile("references-direct", ".bin");
        try {
            source.writeBinary(bin);
            java.util.Random rnd = new java.util.Random(42);
            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                for (int iter = 0; iter < 200; iter++) {
                    double[] probe = new double[14];
                    for (int d = 0; d < 14; d++) {
                        probe[d] = rnd.nextDouble() * 2 - 1; // [-1,1]
                    }
                    short[] q = ReferenceDataset.quantize(probe);
                    for (int i = 0; i < mapped.count(); i++) {
                        assertEquals(mapped.squaredDistance(q, i),
                                mapped.squaredDistanceDirect(q, i),
                                "direct-read mismatch at record " + i + " iter " + iter);
                    }
                }
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void streamsExampleReferencesFixture() throws Exception {
        try (InputStream in = getClass().getResourceAsStream("/example-references.json")) {
            ReferenceDataset dataset = ReferenceDataset.load(in);
            assertEquals(100, dataset.count());
        }
    }

    @Test
    void parsesVectorAndLabel() throws Exception {
        String json = """
                [
                  { "vector": [0,0,0,0,0,-1,-1,0,0,0,0,0,0,0], "label": "legit" },
                  { "vector": [1,1,1,1,1,1,1,1,1,1,1,1,1,1], "label": "fraud" }
                ]
                """;
        ReferenceDataset dataset = ReferenceDataset.load(stream(json));

        assertEquals(2, dataset.count());
        assertTrue(dataset.isFraud(1));
        assertEquals(false, dataset.isFraud(0));
        double[] query = new double[14];
        query[5] = -1;
        query[6] = -1;
        assertEquals(0L, dataset.squaredDistance(ReferenceDataset.quantize(query), 0));
    }

    @Test
    void gunzipsGzFilesTransparently() throws Exception {
        String json = "[ { \"vector\": [0,0,0,0,0,0,0,0,0,0,0,0,0,0], \"label\": \"legit\" } ]";
        Path tmp = Files.createTempFile("refs", ".json.gz");
        try {
            ByteArrayOutputStream buffer = new ByteArrayOutputStream();
            try (GZIPOutputStream gz = new GZIPOutputStream(buffer)) {
                gz.write(json.getBytes(StandardCharsets.UTF_8));
            }
            Files.write(tmp, buffer.toByteArray());

            ReferenceDataset dataset = ReferenceDataset.loadFromFile(tmp);
            assertEquals(1, dataset.count());
        } finally {
            Files.deleteIfExists(tmp);
        }
    }

    @Test
    void rejectsVectorWithWrongDimensionCount() {
        String json = "[ { \"vector\": [1,2,3], \"label\": \"legit\" } ]";
        IOException e = assertThrows(IOException.class, () -> ReferenceDataset.load(stream(json)));
        assertTrue(e.getMessage().contains("dimensions"));
    }

    @Test
    void rejectsRecordMissingVector() {
        String json = "[ { \"label\": \"fraud\" } ]";
        assertThrows(IOException.class, () -> ReferenceDataset.load(stream(json)));
    }

    @Test
    void quantizationPreservesSentinelGrouping() throws Exception {
        // Two references: one "no history" (-1 sentinel), one "has history".
        String json = """
                [
                  { "vector": [0.5,0.5,0.5,0.5,0.5,-1,-1,0.5,0.5,0,0,0,0.5,0.5], "label": "legit" },
                  { "vector": [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0.5,0.5], "label": "legit" }
                ]
                """;
        ReferenceDataset dataset = ReferenceDataset.load(stream(json));

        double[] noHistoryQuery = {0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 0, 0, 0.5, 0.5};
        short[] q = ReferenceDataset.quantize(noHistoryQuery);

        long distToNoHistory = dataset.squaredDistance(q, 0);
        long distToHasHistory = dataset.squaredDistance(q, 1);

        assertEquals(0L, distToNoHistory);
        // sentinel (-1) vs 0.5 on dims 5 and 6: 2 * (1.5 * 10000)^2.
        assertEquals(2L * 15000L * 15000L, distToHasHistory);
        assertTrue(distToHasHistory > distToNoHistory);
    }

    @Test
    void writeBinaryThenMmapRoundTrips() throws Exception {
        ReferenceDataset source;
        try (InputStream in = getClass().getResourceAsStream("/example-references.json")) {
            source = ReferenceDataset.load(in);
        }

        Path bin = Files.createTempFile("references", ".bin");
        try {
            source.writeBinary(bin);
            double[] probe = new double[14];
            short[] q = ReferenceDataset.quantize(probe);

            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                assertEquals(source.count(), mapped.count());
                for (int i = 0; i < source.count(); i++) {
                    assertEquals(source.isFraud(i), mapped.isFraud(i), "label mismatch at " + i);
                    assertEquals(source.squaredDistance(q, i), mapped.squaredDistance(q, i),
                            "distance mismatch at " + i);
                }
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void writeBinaryV2ThenMmapExposesIndex() throws Exception {
        ReferenceDataset source;
        try (InputStream in = getClass().getResourceAsStream("/example-references.json")) {
            source = ReferenceDataset.load(in);
        }

        IvfIndexBuilder.Result index = IvfIndexBuilder.build(source, 8, 6);
        Path bin = Files.createTempFile("references-v2", ".bin");
        try {
            ReferenceDataset.writeBinaryV2(bin, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());

            try (ReferenceDataset mapped = ReferenceDataset.mmap(bin)) {
                assertTrue(mapped.hasIndex());
                assertEquals(8, mapped.clusters());
                assertEquals(source.count(), mapped.count());
                assertEquals(0, mapped.clusterStart(0));
                assertEquals(source.count(), mapped.clusterEnd(mapped.clusters() - 1));

                int reordered = 0;
                for (int c = 0; c < mapped.clusters(); c++) {
                    reordered += mapped.clusterEnd(c) - mapped.clusterStart(c);
                }
                assertEquals(source.count(), reordered);
            }
        } finally {
            Files.deleteIfExists(bin);
        }
    }

    @Test
    void mmapRejectsBadMagic() throws Exception {
        Path bogus = Files.createTempFile("bogus", ".bin");
        try {
            Files.write(bogus, new byte[64]);
            assertThrows(IOException.class, () -> ReferenceDataset.mmap(bogus));
        } finally {
            Files.deleteIfExists(bogus);
        }
    }

    private static InputStream stream(String json) {
        return new ByteArrayInputStream(json.getBytes(StandardCharsets.UTF_8));
    }
}
