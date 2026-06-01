package io.github.brunoborges.rinha2026;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.zip.GZIPInputStream;

/**
 * Immutable, off-heap store of the labeled reference vectors the vector search
 * queries (see {@code docs/DATASET.md}).
 *
 * <p>Inspired by the memory-bandwidth playbook of the 1BRC winners, the dataset
 * is:
 * <ul>
 *   <li><b>Quantized.</b> Each {@code [0,1]} value (and the {@code -1} sentinel)
 *       is stored as an {@code int16} scaled by {@value #SCALE}. Because the
 *       reference vectors carry four decimals, this is <i>lossless</i> for them,
 *       halves the footprint versus {@code float}, and lets distances be computed
 *       with cheap integer math.</li>
 *   <li><b>Off-heap.</b> Vectors live in a {@link MemorySegment}, not the Java
 *       heap, so the GC never walks them.</li>
 *   <li><b>Memory-mappable.</b> {@link #mmap(Path)} maps a pre-built
 *       {@code .bin} file read-only; two API instances mapping the same file
 *       share one set of OS page-cache pages, so the data counts once against
 *       the challenge's memory budget.</li>
 * </ul>
 *
 * <h2>Binary format ({@code .bin})</h2>
 * Little-endian throughout:
 * <pre>
 *   offset  size  field
 *   0       4     magic 'R','I','N','1'
 *   4       4     version (int, 1 or 2)
 *   8       4     dimensions (int, = 14)
 *   12      4     scale (int, = 10000)
 *   16      8     count (long)
 *   24      4     clusters (int; 0 when no IVF index)
 *   28      4     reserved
 * </pre>
 *
 * <h3>Version 1 (no index)</h3>
 * <pre>
 *   32      count * dimensions * 2   int16 vectors (row-major)
 *   ...     count                    label bytes (1 = fraud, 0 = legit)
 * </pre>
 *
 * <h3>Version 2 (IVF index)</h3>
 * Vectors and labels are reordered so each cluster is contiguous; an inverted
 * file lets a query scan only a few probed clusters instead of all records.
 * <pre>
 *   32      clusters * dimensions * 2  int16 centroids (row-major)
 *   ...     (clusters + 1) * 4         int32 CSR offsets (offsets[clusters] == count)
 *   ...     count * dimensions * 2     int16 vectors (reordered, row-major)
 *   ...     count                      label bytes (reordered)
 * </pre>
 */
public final class ReferenceDataset implements AutoCloseable {

    /** Fixed-point scale applied to every dimension. */
    public static final int SCALE = 10_000;

    private static final int DIMS = TransactionVectorizer.DIMENSIONS;
    private static final int HEADER_BYTES = 32;
    private static final int FORMAT_VERSION = 1;
    private static final int FORMAT_VERSION_INDEXED = 2;
    private static final byte[] MAGIC = {'R', 'I', 'N', '1'};

    /** Little-endian, unaligned int16 access shared by every backing segment. */
    private static final ValueLayout.OfShort SHORT_LE =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final MemorySegment vectors;
    private final MemorySegment labels;
    private final int count;
    private final Arena arena;

    /** IVF index (version 2 only); {@code null}/empty when absent. */
    private final MemorySegment centroids;
    private final int[] clusterOffsets;
    private final int clusters;

    /**
     * Small hot tables copied onto the Java heap at load. GraalVM native-image
     * does <em>not</em> intrinsify per-element {@link MemorySegment} reads the
     * way HotSpot C2 does, so reading int16/byte values one at a time from the
     * off-heap segment costs ~200&nbsp;ns each &mdash; ~100&times; slower than a
     * heap array load and enough to collapse the service under load. The
     * centroid table (a few tens of KiB) and the labels (one byte per record)
     * are tiny, so we keep them on heap and let the JIT/AOT compiler emit plain,
     * vectorizable array accesses. The 84&nbsp;MiB vector table stays off-heap
     * (it would not fit the heap budget) and is read via bulk
     * {@link #copyVectorRange} copies instead.
     */
    private final short[] centroidsHeap;
    private final byte[] labelsHeap;

    private ReferenceDataset(MemorySegment vectors, MemorySegment labels, int count, Arena arena) {
        this(vectors, labels, count, arena, null, null, 0);
    }

    private ReferenceDataset(MemorySegment vectors, MemorySegment labels, int count, Arena arena,
                             MemorySegment centroids, int[] clusterOffsets, int clusters) {
        this.vectors = vectors;
        this.labels = labels;
        this.count = count;
        this.arena = arena;
        this.centroids = centroids;
        this.clusterOffsets = clusterOffsets;
        this.clusters = clusters;
        this.labelsHeap = copyLabelsToHeap(labels, count);
        this.centroidsHeap = clusters > 0 && centroids != null
                ? copyCentroidsToHeap(centroids, clusters)
                : new short[0];
    }

    private static byte[] copyLabelsToHeap(MemorySegment labels, int count) {
        byte[] out = new byte[count];
        MemorySegment.copy(labels, ValueLayout.JAVA_BYTE, 0, out, 0, count);
        return out;
    }

    private static short[] copyCentroidsToHeap(MemorySegment centroids, int clusters) {
        short[] out = new short[clusters * DIMS];
        MemorySegment.copy(centroids, SHORT_LE, 0, out, 0, clusters * DIMS);
        return out;
    }

    /** @return the number of reference records */
    public int count() {
        return count;
    }

    /** @return whether this dataset carries an IVF index (format version 2). */
    public boolean hasIndex() {
        return clusters > 0;
    }

    /** @return the number of IVF clusters, or {@code 0} when there is no index. */
    public int clusters() {
        return clusters;
    }

    /** @return the first record index (inclusive) of cluster {@code c}. */
    public int clusterStart(int c) {
        return clusterOffsets[c];
    }

    /** @return the last record index (exclusive) of cluster {@code c}. */
    public int clusterEnd(int c) {
        return clusterOffsets[c + 1];
    }

    /** Sink that prevents the JIT from eliding the {@link #preload()} reads. */
    private static long preloadSink;

    /**
     * Touches every page of the memory-mapped dataset (vectors, labels and, when
     * present, the IVF centroids) so the OS faults them into the page cache up
     * front. Without this, the first burst of live traffic pays the major page
     * faults itself &mdash; and under the challenge's single-CPU budget those
     * synchronous faults back up the request queue and produce a wave of errors
     * before the cache warms. Calling this during startup (before the port opens)
     * keeps {@code GET /ready} a truthful "served at full speed" signal.
     *
     * <p>Reads one {@code byte} per 4&nbsp;KiB page into a {@code static} sink so
     * the loads cannot be optimized away. A no-op for in-memory datasets, whose
     * pages are already resident.
     */
    public void preload() {
        if (arena == null) {
            return;
        }
        // Centroids and labels were already faulted in (and copied to the heap)
        // at load; only the off-heap vector table still needs pre-faulting.
        final int pageSize = 4096;
        preloadSink += touch(vectors, pageSize);
    }

    private static long touch(MemorySegment segment, int pageSize) {
        long size = segment.byteSize();
        long sum = 0;
        for (long off = 0; off < size; off += pageSize) {
            sum += segment.get(ValueLayout.JAVA_BYTE, off);
        }
        if (size > 0) {
            sum += segment.get(ValueLayout.JAVA_BYTE, size - 1);
        }
        return sum;
    }

    /**
     * Squared Euclidean distance, in quantized units, between a quantized query
     * and the IVF centroid {@code c}. Same arithmetic as
     * {@link #squaredDistance(short[], int)} but over the centroid table.
     */
    public long centroidSquaredDistance(short[] query, int c) {
        int base = c * DIMS;
        long sum0 = 0;
        long sum1 = 0;
        for (int d = 0; d < DIMS; d += 2) {
            int a = query[d] - centroidsHeap[base + d];
            int b = query[d + 1] - centroidsHeap[base + d + 1];
            sum0 += (long) a * a;
            sum1 += (long) b * b;
        }
        return sum0 + sum1;
    }

    /** @return whether the record at {@code index} is labeled fraud */
    public boolean isFraud(int index) {
        return labelsHeap[index] != 0;
    }

    /**
     * Bulk-copies {@code countVec} reordered vectors starting at record
     * {@code startVec} into {@code dst} (row-major, {@code countVec * DIMENSIONS}
     * shorts from {@code dstOff}). This is the throughput-critical accessor: a
     * single bulk {@link MemorySegment#copy} moves the off-heap int16 data onto
     * the heap so the distance loop can run over plain (intrinsified) array
     * accesses instead of per-element FFM reads, which GraalVM native-image does
     * not intrinsify. {@code dst} must hold at least {@code dstOff + countVec * DIMENSIONS}.
     */
    public void copyVectorRange(int startVec, int countVec, short[] dst, int dstOff) {
        long srcByteOffset = (long) startVec * DIMS * Short.BYTES;
        MemorySegment.copy(vectors, SHORT_LE, srcByteOffset, dst, dstOff, countVec * DIMS);
    }

    /**
     * Squared Euclidean distance, in quantized units, between a quantized query
     * and the record at {@code index}. Squared distance is enough for
     * nearest-neighbor ranking and skips a {@code sqrt} per comparison. Two
     * partial accumulators break the serial dependency chain so the CPU can
     * pipeline the multiply-adds.
     *
     * @param query a quantized query of length {@link TransactionVectorizer#DIMENSIONS}
     * @param index the reference record index
     * @return the squared distance (fits comfortably in a {@code long})
     */
    public long squaredDistance(short[] query, int index) {
        long base = (long) index * DIMS;
        long sum0 = 0;
        long sum1 = 0;
        for (int d = 0; d < DIMS; d += 2) {
            int a = query[d] - vectors.getAtIndex(SHORT_LE, base + d);
            int b = query[d + 1] - vectors.getAtIndex(SHORT_LE, base + d + 1);
            sum0 += (long) a * a;
            sum1 += (long) b * b;
        }
        return sum0 + sum1;
    }

    /**
     * Quantizes one normalized value to its {@code int16} fixed-point form.
     * Values are clamped to {@code [-1, 1]} (real dimensions are already in
     * {@code [0,1]}; the only legal negative is the {@code -1} sentinel).
     */
    public static short quantize(double value) {
        double clamped = value < -1.0 ? -1.0 : Math.min(value, 1.0);
        return (short) Math.round(clamped * SCALE);
    }

    /** Quantizes a full normalized vector. */
    public static short[] quantize(double[] vector) {
        short[] q = new short[vector.length];
        for (int i = 0; i < vector.length; i++) {
            q[i] = quantize(vector[i]);
        }
        return q;
    }

    /**
     * Copies every quantized vector into a flat heap {@code short[]} of
     * {@code count * DIMENSIONS} (row-major). Used by the offline
     * {@link IvfIndexBuilder}; not on any request path.
     */
    short[] toQuantizedArray() {
        short[] out = new short[count * DIMS];
        MemorySegment.copy(vectors, SHORT_LE, 0, out, 0, count * DIMS);
        return out;
    }

    /** Copies every label into a heap {@code byte[]} (1 = fraud). Build-time only. */
    byte[] toLabelArray() {
        byte[] out = new byte[count];
        MemorySegment.copy(labels, ValueLayout.JAVA_BYTE, 0, out, 0, count);
        return out;
    }

    /**
     * Builds an in-memory dataset from flat {@code float} vectors. Primarily for
     * tests; the segment is auto-managed (freed by the GC).
     *
     * @param flatVectors {@code count * DIMENSIONS} normalized values, row-major
     * @param fraud       per-record fraud label
     * @param count       number of records
     */
    public static ReferenceDataset of(float[] flatVectors, boolean[] fraud, int count) {
        if (flatVectors.length < count * DIMS) {
            throw new IllegalArgumentException("vectors array too small for count");
        }
        if (fraud.length < count) {
            throw new IllegalArgumentException("fraud array too small for count");
        }
        Arena arena = Arena.ofAuto();
        MemorySegment vectors = arena.allocate((long) count * DIMS * Short.BYTES);
        MemorySegment labels = arena.allocate(count);
        for (int i = 0; i < count; i++) {
            long base = (long) i * DIMS;
            for (int d = 0; d < DIMS; d++) {
                vectors.setAtIndex(SHORT_LE, base + d, quantize(flatVectors[i * DIMS + d]));
            }
            labels.set(ValueLayout.JAVA_BYTE, i, (byte) (fraud[i] ? 1 : 0));
        }
        return new ReferenceDataset(vectors, labels, count, null);
    }

    /**
     * Memory-maps a pre-built {@code .bin} file read-only. This is the preferred
     * production path: no parsing, off-heap, and shareable across processes via
     * the OS page cache.
     *
     * @param binFile a file produced by {@link #writeBinary(Path)} / {@link ReferenceConverter}
     * @return the mapped dataset; {@link #close()} unmaps it
     * @throws IOException if the file cannot be mapped or has a bad header
     */
    public static ReferenceDataset mmap(Path binFile) throws IOException {
        Arena arena = Arena.ofShared();
        try (FileChannel channel = FileChannel.open(binFile, StandardOpenOption.READ)) {
            long size = channel.size();
            if (size < HEADER_BYTES) {
                throw new IOException("file too small to be a reference dataset: " + binFile);
            }
            MemorySegment file = channel.map(FileChannel.MapMode.READ_ONLY, 0, size, arena);

            ByteBuffer header = file.asSlice(0, HEADER_BYTES).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN);
            byte[] magic = new byte[MAGIC.length];
            header.get(magic);
            if (!Arrays.equals(magic, MAGIC)) {
                throw new IOException("bad magic; not a reference dataset: " + binFile);
            }
            int version = header.getInt();
            int dims = header.getInt();
            int scale = header.getInt();
            long count = header.getLong();
            int clusters = header.getInt();
            if (version != FORMAT_VERSION && version != FORMAT_VERSION_INDEXED) {
                throw new IOException("unsupported format version " + version);
            }
            if (dims != DIMS) {
                throw new IOException("dimension mismatch: file has " + dims + ", expected " + DIMS);
            }
            if (scale != SCALE) {
                throw new IOException("scale mismatch: file has " + scale + ", expected " + SCALE);
            }
            if (count < 0 || count > Integer.MAX_VALUE) {
                throw new IOException("unsupported record count: " + count);
            }
            if (version == FORMAT_VERSION) {
                clusters = 0;
            }
            if (clusters < 0 || clusters > Integer.MAX_VALUE / DIMS) {
                throw new IOException("unsupported cluster count: " + clusters);
            }

            long vectorsBytes = count * DIMS * Short.BYTES;
            if (clusters == 0) {
                long expected = HEADER_BYTES + vectorsBytes + count;
                if (size < expected) {
                    throw new IOException("truncated dataset: expected " + expected + " bytes, got " + size);
                }
                MemorySegment vectors = file.asSlice(HEADER_BYTES, vectorsBytes);
                MemorySegment labels = file.asSlice(HEADER_BYTES + vectorsBytes, count);
                return new ReferenceDataset(vectors, labels, (int) count, arena);
            }

            long centroidsBytes = (long) clusters * DIMS * Short.BYTES;
            long offsetsBytes = (long) (clusters + 1) * Integer.BYTES;
            long expected = HEADER_BYTES + centroidsBytes + offsetsBytes + vectorsBytes + count;
            if (size < expected) {
                throw new IOException("truncated indexed dataset: expected " + expected
                        + " bytes, got " + size);
            }
            long pos = HEADER_BYTES;
            MemorySegment centroids = file.asSlice(pos, centroidsBytes);
            pos += centroidsBytes;
            int[] clusterOffsets = new int[clusters + 1];
            file.asSlice(pos, offsetsBytes).asByteBuffer().order(ByteOrder.LITTLE_ENDIAN)
                    .asIntBuffer().get(clusterOffsets);
            pos += offsetsBytes;
            if (clusterOffsets[0] != 0 || clusterOffsets[clusters] != count) {
                throw new IOException("invalid CSR offsets: first=" + clusterOffsets[0]
                        + ", last=" + clusterOffsets[clusters] + ", count=" + count);
            }
            MemorySegment vectors = file.asSlice(pos, vectorsBytes);
            pos += vectorsBytes;
            MemorySegment labels = file.asSlice(pos, count);
            return new ReferenceDataset(vectors, labels, (int) count, arena,
                    centroids, clusterOffsets, clusters);
        } catch (IOException | RuntimeException e) {
            arena.close();
            throw e;
        }
    }

    /**
     * Writes this dataset to {@code binFile} in the binary format documented on
     * this class. Run once offline so the server only ever {@link #mmap(Path) mmaps}.
     *
     * @param binFile destination path (overwritten if present)
     * @throws IOException on write failure
     */
    public void writeBinary(Path binFile) throws IOException {
        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.put(MAGIC);
        header.putInt(FORMAT_VERSION);
        header.putInt(DIMS);
        header.putInt(SCALE);
        header.putLong(count);
        header.putInt(0); // clusters (no index)
        header.putInt(0); // reserved
        header.flip();

        try (FileChannel channel = FileChannel.open(binFile,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writeFully(channel, header);
            writeFully(channel, vectors.asSlice(0, (long) count * DIMS * Short.BYTES).asByteBuffer());
            writeFully(channel, labels.asSlice(0, count).asByteBuffer());
        }
    }

    /**
     * Writes an IVF-indexed (version 2) {@code .bin}. Vectors and labels must be
     * reordered so each cluster is contiguous, described by CSR {@code offsets}
     * (length {@code clusters + 1}, {@code offsets[0] == 0},
     * {@code offsets[clusters] == count}). Run offline by {@link IvfIndexBuilder}.
     *
     * @param binFile   destination path (overwritten if present)
     * @param count     number of records
     * @param vectors   {@code count * DIMENSIONS} reordered int16 vectors (row-major)
     * @param labels    {@code count} reordered label bytes
     * @param centroids {@code clusters * DIMENSIONS} int16 centroids (row-major)
     * @param offsets   CSR cluster offsets, length {@code clusters + 1}
     * @throws IOException on write failure
     */
    public static void writeBinaryV2(Path binFile, int count, short[] vectors, byte[] labels,
                                     short[] centroids, int[] offsets) throws IOException {
        int clusters = offsets.length - 1;
        if (vectors.length != count * DIMS) {
            throw new IllegalArgumentException("vectors length != count * DIMS");
        }
        if (labels.length != count) {
            throw new IllegalArgumentException("labels length != count");
        }
        if (centroids.length != clusters * DIMS) {
            throw new IllegalArgumentException("centroids length != clusters * DIMS");
        }
        if (offsets[0] != 0 || offsets[clusters] != count) {
            throw new IllegalArgumentException("offsets must start at 0 and end at count");
        }

        ByteBuffer header = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        header.put(MAGIC);
        header.putInt(FORMAT_VERSION_INDEXED);
        header.putInt(DIMS);
        header.putInt(SCALE);
        header.putLong(count);
        header.putInt(clusters);
        header.putInt(0); // reserved
        header.flip();

        ByteBuffer centroidBuf = ByteBuffer.allocate(centroids.length * Short.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        centroidBuf.asShortBuffer().put(centroids);
        ByteBuffer offsetBuf = ByteBuffer.allocate(offsets.length * Integer.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        offsetBuf.asIntBuffer().put(offsets);
        ByteBuffer vectorBuf = ByteBuffer.allocate(vectors.length * Short.BYTES)
                .order(ByteOrder.LITTLE_ENDIAN);
        vectorBuf.asShortBuffer().put(vectors);
        ByteBuffer labelBuf = ByteBuffer.wrap(labels);

        try (FileChannel channel = FileChannel.open(binFile,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writeFully(channel, header);
            writeFully(channel, centroidBuf);
            writeFully(channel, offsetBuf);
            writeFully(channel, vectorBuf);
            writeFully(channel, labelBuf);
        }
    }

    private static void writeFully(FileChannel channel, ByteBuffer buffer) throws IOException {
        while (buffer.hasRemaining()) {
            channel.write(buffer);
        }
    }

    /**
     * Loads a dataset from a JSON file (gunzipping {@code .gz}). This is the
     * fallback when no {@code .bin} exists; it still ends up off-heap, but pays
     * the streaming-parse cost. Prefer building a {@code .bin} and {@link #mmap}.
     *
     * @param path a {@code .json} or {@code .json.gz} file
     * @return the loaded dataset
     * @throws IOException if the file cannot be read or is malformed
     */
    public static ReferenceDataset loadFromFile(Path path) throws IOException {
        try (InputStream raw = new BufferedInputStream(Files.newInputStream(path), 1 << 16)) {
            InputStream in = path.getFileName().toString().endsWith(".gz")
                    ? new GZIPInputStream(raw, 1 << 16)
                    : raw;
            return load(in);
        }
    }

    /**
     * Streams reference records from a JSON array of
     * {@code {"vector":[...14...],"label":"fraud"|"legit"}} objects, quantizing
     * into an off-heap segment.
     *
     * @param in the JSON input stream (already decompressed)
     * @return the loaded dataset
     * @throws IOException if the stream cannot be read or is malformed
     */
    public static ReferenceDataset load(InputStream in) throws IOException {
        short[] values = new short[DIMS * 1024];
        byte[] fraud = new byte[1024];
        int count = 0;

        JsonFactory factory = new JsonFactory();
        try (JsonParser p = factory.createParser(in)) {
            if (p.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected a JSON array at the top level");
            }
            while (p.nextToken() == JsonToken.START_OBJECT) {
                if ((long) count * DIMS + DIMS > values.length) {
                    values = Arrays.copyOf(values, Math.max(values.length << 1, (count + 1) * DIMS));
                }
                if (count + 1 > fraud.length) {
                    fraud = Arrays.copyOf(fraud, Math.max(fraud.length << 1, count + 1));
                }

                boolean haveVector = false;
                int base = count * DIMS;
                while (p.nextToken() == JsonToken.FIELD_NAME) {
                    String field = p.currentName();
                    p.nextToken();
                    if ("vector".equals(field)) {
                        readVector(p, values, base);
                        haveVector = true;
                    } else if ("label".equals(field)) {
                        fraud[count] = (byte) ("fraud".equals(p.getValueAsString()) ? 1 : 0);
                    } else {
                        p.skipChildren();
                    }
                }
                if (!haveVector) {
                    throw new IOException("record " + count + " is missing 'vector'");
                }
                count++;
            }
        }

        Arena arena = Arena.ofAuto();
        MemorySegment vectors = arena.allocate((long) count * DIMS * Short.BYTES);
        MemorySegment.copy(values, 0, vectors, SHORT_LE, 0, count * DIMS);
        MemorySegment labels = arena.allocate(count);
        MemorySegment.copy(fraud, 0, labels, ValueLayout.JAVA_BYTE, 0, count);
        return new ReferenceDataset(vectors, labels, count, null);
    }

    private static void readVector(JsonParser p, short[] values, int base) throws IOException {
        if (p.currentToken() != JsonToken.START_ARRAY) {
            throw new IOException("'vector' must be an array");
        }
        int d = 0;
        while (p.nextToken() != JsonToken.END_ARRAY) {
            if (d >= DIMS) {
                throw new IOException("'vector' has more than " + DIMS + " dimensions");
            }
            values[base + d] = quantize(p.getDoubleValue());
            d++;
        }
        if (d != DIMS) {
            throw new IOException("'vector' has " + d + " dimensions, expected " + DIMS);
        }
    }

    /** Unmaps the backing file (mmap datasets); a no-op for in-memory datasets. */
    @Override
    public void close() {
        if (arena != null) {
            arena.close();
        }
    }
}
