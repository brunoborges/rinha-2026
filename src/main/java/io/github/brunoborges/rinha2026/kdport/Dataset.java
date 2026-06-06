package io.github.brunoborges.rinha2026.kdport;

import io.github.brunoborges.rinha2026.ReferenceDataset;

/** Minimal dataset shim for the offline kdport experiment. */
public final class Dataset {
    public static final int DIMS = 14;
    public static final int STRIDE = 16;

    private final int size;
    private final float[] vectors;
    private final boolean[] fraudLabels;

    public Dataset(int size, float[] vectors, boolean[] fraudLabels) {
        if (size < 0) {
            throw new IllegalArgumentException("size must be non-negative");
        }
        if (vectors.length < size * STRIDE) {
            throw new IllegalArgumentException("vectors length must be at least size*16");
        }
        if (fraudLabels.length < size) {
            throw new IllegalArgumentException("fraudLabels length must be at least size");
        }
        this.size = size;
        this.vectors = vectors;
        this.fraudLabels = fraudLabels;
    }

    public int size() {
        return size;
    }

    public float[] vectors() {
        return vectors;
    }

    public boolean[] fraudLabels() {
        return fraudLabels;
    }

    public static Dataset fromReferenceDataset(ReferenceDataset ds) {
        int n = ds.count();
        short[] vec16 = ds.toQuantizedArray();
        byte[] labels = ds.toLabelArray();
        float[] vectors = new float[n * STRIDE];
        boolean[] fraud = new boolean[n];
        for (int i = 0; i < n; i++) {
            int src = i * DIMS;
            int dst = i * STRIDE;
            for (int d = 0; d < DIMS; d++) {
                vectors[dst + d] = vec16[src + d] / 10000.0f;
            }
            fraud[i] = labels[i] != 0;
        }
        return new Dataset(n, vectors, fraud);
    }
}
