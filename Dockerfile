# syntax=docker/dockerfile:1

# =============================================================================
# Rinha de Backend 2026 — fraud-scoring API (GraalVM 25 Native Image)
#
# Three-stage build:
#   1. build    — maven + Temurin JDK 25 (Debian). Compile the shaded jar,
#                 download the 3M reference dataset and pre-build the off-heap
#                 int16 IVF `references.bin`.
#   2. native   — GraalVM 25 native-image. Ahead-of-time compile the jar into a
#                 standalone native executable (NO network here: Oracle Linux's
#                 OpenSSL breaks `curl`, but native-image needs no downloads).
#   3. runtime  — Oracle Linux 10 slim (glibc matches the GraalVM builder) + the
#                 native binary + references.bin. No JDK, no JIT warmup.
#
# Why native image: the eval starts containers cold and scores p99 hard. AOT
# removes JIT warmup entirely (the server runs at full speed immediately), so all
# the previous warmup code is gone — the only pre-serving step left is faulting
# the mmap'd dataset into the page cache (App.preload), an OS concern AOT does
# NOT solve.
#
# The dataset is converted at BUILD time (docs/EVALUATION.md: "the more
# processing you move outside of runtime, the better your p99"). Nothing parses
# the dataset JSON at runtime; request parsing is reflection-free jackson-core
# streaming, so the native image needs no reachability metadata.
# =============================================================================

# ---- 1. build stage: shaded jar + IVF references.bin ------------------------
# Debian-based: its curl/TLS works (unlike the Oracle Linux GraalVM image), and
# mvn resolves dependencies over Java's own TLS stack.
FROM maven:3.9-eclipse-temurin-25 AS build

# Public mirror of the 3,000,000-record labelled reference dataset.
ARG REFERENCES_URL=https://raw.githubusercontent.com/brunoborges/rinha-de-backend-2026/main/resources/references.json.gz

WORKDIR /src

# Maven config first (better layer caching for dependency resolution).
COPY .mvn ./.mvn
COPY mvnw pom.xml ./

# Warm the dependency cache.
RUN mvn -B -q dependency:go-offline || true

COPY src ./src

# Build the shaded jar. databind/jsr310 are test-scoped, so the jar is
# reflection-free (only jackson-core streaming + jackson-annotations). Tests run
# on the host; skip them here.
RUN mvn -B -DskipTests package

# Pre-build the binary reference dataset (offline JSON -> off-heap int16 .bin)
# WITH an IVF (inverted-file) ANN index baked in (version-2 format): k-means
# clusters the 3M vectors so each query scans only a few clusters at runtime
# instead of the whole dataset. Runs on all build cores; runtime stays at 1 CPU.
RUN set -eux; \
    curl -fSL -o /tmp/references.json.gz "$REFERENCES_URL"; \
    mkdir -p /app; \
    java --enable-native-access=ALL-UNNAMED \
        -cp target/rinha-2026-1.0-SNAPSHOT.jar \
        io.github.brunoborges.rinha2026.ReferenceConverter \
        /tmp/references.json.gz /app/references.bin; \
    rm -f /tmp/references.json.gz; \
    ls -la /app/references.bin

# ---- 2. native stage: ahead-of-time compile ---------------------------------
FROM container-registry.oracle.com/graalvm/native-image:25 AS native

WORKDIR /src
COPY --from=build /src/target/rinha-2026-1.0-SNAPSHOT.jar ./app.jar

# Native Image configuration
RUN mkdir -p /app; \
    native-image \
        --no-fallback \
        --gc=G1 \
        -march=native \
        --enable-native-access=ALL-UNNAMED \
        --add-modules=jdk.httpserver \
        -O3 \
        -H:TuneInlinerExploration=1 \
        -R:MaxHeapSize=64m \
        -jar app.jar \
        /app/rinha-app; \
    ls -la /app/rinha-app

# ---- 3. runtime stage -------------------------------------------------------
# Oracle Linux 10 slim: same glibc family as the GraalVM 25 builder, so the
# dynamically-linked native binary loads cleanly (a Debian/distroless base can
# carry an older glibc and fail with GLIBC_x.y not found).
FROM container-registry.oracle.com/os/oraclelinux:10-slim AS runtime

RUN useradd --system --uid 10001 --no-create-home app

WORKDIR /app
COPY --from=native /app/rinha-app /app/rinha-app
COPY --from=build /app/references.bin /app/references.bin

# The native binary reads these at runtime (System.getenv). NPROBE / SCAN_CAP
# tune the IVF search (clusters probed per query / max records scanned); defaults
# match brute-force decisions exactly while cutting per-query work ~50x. WORKERS
# sizes the per-instance parse/scan pools — concurrent scans overlap memory
# stalls within the 0.45 CPU quota (availableProcessors() is pinned to 1 by the
# cgroup). Heap is baked into the binary (-R: flags above) — no JAVA_OPTS.
ENV REFERENCES_BIN=/app/references.bin \
    PORT=8080 \
    NPROBE=24 \
    SCAN_CAP=120000 \
    WORKERS=4

USER app
EXPOSE 8080

HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=20 \
    CMD curl -fsS "http://localhost:${PORT}/ready" || exit 1

ENTRYPOINT ["/app/rinha-app"]
