package io.github.brunoborges.rinha2026;

import io.github.brunoborges.rinha2026.kdport.Dataset;
import io.github.brunoborges.rinha2026.kdport.KdTree;
import io.github.brunoborges.rinha2026.kdport.KdTreeBuilder;
import io.github.brunoborges.rinha2026.kdport.KdTreeIO;

import java.nio.file.Path;

/**
 * Offline tool: builds the {@link KdTree} index binary (mmap-loadable at runtime via
 * {@link KdTreeIO#loadMmap}) from the off-heap int16 reference dataset.
 *
 * <p>Run at image-build time only (never on a request path). Needs a large heap for the
 * transient float source vectors over 3M points (e.g. {@code -Xmx3g}).
 *
 * <pre>java -Xmx3g -cp app.jar io.github.brunoborges.rinha2026.KdTreeIndexBuilder references.bin kdtree.bin</pre>
 */
public final class KdTreeIndexBuilder {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: KdTreeIndexBuilder <references.bin> <kdtree.bin>");
            System.exit(2);
        }
        Path in = Path.of(args[0]);
        Path out = Path.of(args[1]);

        long t0 = System.nanoTime();
        try (ReferenceDataset ds = ReferenceDataset.mmap(in)) {
            ds.preload();
            int n = ds.count();
            System.out.printf("loaded %,d reference vectors from %s%n", n, in);

            Dataset dataset = Dataset.fromReferenceDataset(ds);
            long t1 = System.nanoTime();
            System.out.printf("dequantized to f32 source in %d ms%n", (t1 - t0) / 1_000_000);

            KdTree tree = KdTreeBuilder.build(dataset);
            long t2 = System.nanoTime();
            System.out.printf("built KdTree (%,d nodes) in %d ms%n", tree.size(), (t2 - t1) / 1_000_000);

            KdTreeIO.save(tree, out);
            long t3 = System.nanoTime();
            System.out.printf("wrote %s in %d ms%n", out, (t3 - t2) / 1_000_000);
        }
    }

    private KdTreeIndexBuilder() {
    }
}
