package io.github.brunoborges.rinha2026;

import java.io.IOException;
import java.nio.file.Path;

/**
 * Offline tool that converts the JSON reference dataset
 * ({@code references.json} or {@code references.json.gz}) into the compact
 * memory-mappable {@code .bin} format consumed by {@link ReferenceDataset#mmap}.
 *
 * <p>Run it once as part of building/deploying the image so the server never
 * pays the JSON parsing cost and can share the dataset across instances via the
 * OS page cache:
 *
 * <pre>{@code
 * java -cp app.jar io.github.brunoborges.rinha2026.ReferenceConverter \
 *      resources/references.json.gz resources/references.bin [clusters]
 * }</pre>
 *
 * <p>By default it builds an IVF-indexed (version 2) {@code .bin} so the server
 * uses approximate nearest-neighbor search. Pass {@code 0} as the optional third
 * argument to emit a plain (version 1) file for brute-force scoring instead.
 */
public final class ReferenceConverter {

    private ReferenceConverter() {
    }

    public static void main(String[] args) throws IOException {
        if (args.length < 2 || args.length > 3) {
            System.err.println("Usage: ReferenceConverter <input.json[.gz]> <output.bin> [clusters]");
            System.exit(2);
            return;
        }

        Path input = Path.of(args[0]);
        Path output = Path.of(args[1]);
        int clusters = args.length == 3 ? Integer.parseInt(args[2].trim()) : IvfIndexBuilder.DEFAULT_CLUSTERS;

        long start = System.nanoTime();
        try (ReferenceDataset dataset = ReferenceDataset.loadFromFile(input)) {
            long loadedMs = (System.nanoTime() - start) / 1_000_000;
            System.out.printf("Loaded %,d records from %s in %d ms%n", dataset.count(), input, loadedMs);

            if (clusters <= 0) {
                long writeStart = System.nanoTime();
                dataset.writeBinary(output);
                long writeMs = (System.nanoTime() - writeStart) / 1_000_000;
                System.out.printf("Wrote unindexed %s in %d ms%n", output, writeMs);
                return;
            }

            long buildStart = System.nanoTime();
            IvfIndexBuilder.Result index = IvfIndexBuilder.build(dataset, clusters,
                    IvfIndexBuilder.DEFAULT_ITERATIONS);
            long buildMs = (System.nanoTime() - buildStart) / 1_000_000;
            System.out.printf("Built IVF index (%,d clusters) in %d ms%n", index.clusters(), buildMs);

            long writeStart = System.nanoTime();
            ReferenceDataset.writeBinaryV2(output, index.count(), index.vectors(), index.labels(),
                    index.centroids(), index.offsets());
            long writeMs = (System.nanoTime() - writeStart) / 1_000_000;
            System.out.printf("Wrote indexed %s in %d ms%n", output, writeMs);
        }
    }
}
