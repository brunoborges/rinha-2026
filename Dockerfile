# syntax=docker/dockerfile:1

# =============================================================================
# Rinha de Backend 2026 — fraud-scoring API (HotSpot JDK 27-EA + AOT cache)
#
# Runtime migrated from GraalVM native-image to HotSpot with the Leyden AOT cache
# (JEP 483 class load/link + JEP 515 method profiling). The server runs on OpenJDK
# 27 Early-Access, whose UseCompactObjectHeaders (JEP 519) is on by DEFAULT —
# 8-byte object headers buy heap headroom under the 167 MB cap — and whose C2/AOT
# improvements measurably cut the cold-run p99. The index is now an exact bbox-pruned
# KD-tree (ported from the competitor, MIT): an offline experiment over the 54,100-query
# eval corpus measured E=0 (FP=0, FN=0) at ~2,900 p99 node visits — perfect detection at
# ~10x fewer candidate evaluations than the IVF (~27k for single-digit E). pts is mmap'd
# off-heap (96 MB, reclaimable file cache); topSlot+topBbox (~34 MB) load on-heap. The
# approximate IVF scan is retained as a fallback (SCORER=ivf). The runtime (C2's AVX2 int16
# distance kernel + AOT cache removing JIT warmup + jemalloc + JVM tuning) and the FFM epoll
# fd-passing server (standard on HotSpot; no GraalVM Feature needed) are kept.
#
# Five stages:
#   1. build       — maven + Temurin 25: shaded jar + off-heap int16 IVF references.bin
#                    (--release 25 bytecode + raw data are JDK-version-agnostic).
#   jdk27          — OpenJDK 27-EA tarball base (JAVA_HOME) for the AOT + runtime stages.
#   2. aot-record  — run the scoring hot path (AOT_TRAINING) under -XX:AOTMode=record.
#   3. aot-create  — assemble the AOT cache (-XX:AOTMode=create) from the recorded config.
#   4. runtime     — JDK 27-EA + jemalloc + jar + references.bin + app.aot. A startup
#                    JIT warmup (App.warmup) provokes background C2 and waits for the
#                    compiler to settle before /ready, so the single cold contest run
#                    hits C2-compiled code.
#
# The AOT cache is keyed on the JVM flags, classpath and jar: JVM_FLAGS MUST be
# byte-identical across record, create and the runtime ENTRYPOINT or the cache is
# silently ignored (cold JIT). They are all driven from the single ARG below.
# =============================================================================
# Four stages:
#   1. build       — maven + Temurin 25: shaded jar + off-heap int16 IVF references.bin.
#   2. aot-record  — run the scoring hot path (AOT_TRAINING) under -XX:AOTMode=record.
#   3. aot-create  — assemble the AOT cache (-XX:AOTMode=create) from the recorded config.
#   4. runtime     — Temurin 25 JRE-ish + jemalloc + jar + references.bin + app.aot.
#
# The AOT cache is keyed on the JVM flags, classpath and jar: JVM_FLAGS MUST be
# byte-identical across record, create and the runtime ENTRYPOINT or the cache is
# silently ignored (cold JIT). They are all driven from the single ARG below.
# =============================================================================

# Conservative, cgroup-aware defaults (167 MB / 0.4625 CPU per instance). We do
# NOT copy the competitor's -Xms65m/AlwaysPreTouch/THP bundle blindly: with the
# 84 MB mmap'd dataset that risks OOM on a 167 MB limit. Tune on the VM via the
# JVM_FLAGS build-arg (and the runtime can disable AOT for an A/B by overriding
# AOT_FLAGS="" — flags otherwise identical).
ARG JVM_FLAGS="-XX:+UseSerialGC -Xms24m -Xmx72m -Xss512k -XX:MaxMetaspaceSize=48m -XX:ReservedCodeCacheSize=48m -XX:ActiveProcessorCount=1 -XX:CICompilerCount=2 -XX:+UseFMA -XX:-UsePerfData -XX:+DisableExplicitGC --enable-native-access=ALL-UNNAMED"

# ---- 1. build stage: shaded jar + IVF references.bin ------------------------
FROM maven:3.9-eclipse-temurin-25 AS build

ARG REFERENCES_URL=https://raw.githubusercontent.com/brunoborges/rinha-de-backend-2026/main/resources/references.json.gz

WORKDIR /src
COPY .mvn ./.mvn
COPY mvnw pom.xml ./
RUN mvn -B -q dependency:go-offline || true
COPY src ./src
RUN mvn -B -DskipTests package

# Pre-build the off-heap int16 IVF references.bin (offline; runtime never parses JSON).
RUN set -eux; \
    curl -fSL -o /tmp/references.json.gz "$REFERENCES_URL"; \
    mkdir -p /app; \
    java --enable-native-access=ALL-UNNAMED \
        -cp target/rinha-2026-1.0-SNAPSHOT.jar \
        io.github.brunoborges.rinha2026.ReferenceConverter \
        /tmp/references.json.gz /app/references.bin; \
    rm -f /tmp/references.json.gz; \
    cp target/rinha-2026-1.0-SNAPSHOT.jar /app/app.jar; \
    ls -la /app/references.bin /app/app.jar

# Pre-build the mmap-loadable KdTree index (exact k-NN; offline eval E=0 vs IVF E=26 at ~10x
# fewer node visits). pts is mmap'd off-heap at runtime; topSlot+topBbox (~34 MB) load on-heap.
RUN set -eux; \
    java -Xmx3g --enable-native-access=ALL-UNNAMED \
        -cp target/rinha-2026-1.0-SNAPSHOT.jar \
        io.github.brunoborges.rinha2026.KdTreeIndexBuilder \
        /app/references.bin /app/kdtree.bin; \
    ls -la /app/kdtree.bin

# ---- jdk27: JDK 27 Early-Access runtime (compact object headers) ------------
# JDK 27 EA defaults UseCompactObjectHeaders on (JEP 519), shrinking on-heap
# object headers from 12/16 to 8 bytes — valuable headroom under the 167 MB cap.
# The Maven build above stays on Temurin 25 (it only emits --release 25 bytecode
# + the off-heap references.bin, both JDK-version-agnostic); the AOT cache and the
# server run on JDK 27.
FROM debian:bookworm-slim AS jdk27
ARG JDK27_URL=https://download.java.net/java/early_access/jdk27/24/GPL/openjdk-27-ea+24_linux-x64_bin.tar.gz
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl ca-certificates; \
    curl -fSL -o /tmp/jdk27.tar.gz "$JDK27_URL"; \
    mkdir -p /opt/jdk27; \
    tar -xzf /tmp/jdk27.tar.gz -C /opt/jdk27 --strip-components=1; \
    rm -f /tmp/jdk27.tar.gz; \
    rm -rf /var/lib/apt/lists/*; \
    /opt/jdk27/bin/java -version
ENV JAVA_HOME=/opt/jdk27 PATH=/opt/jdk27/bin:$PATH

# ---- 2. aot-record: capture classes + method profiles -----------------------
FROM jdk27 AS aot-record
ARG JVM_FLAGS
WORKDIR /app
COPY --from=build /app/app.jar /app/app.jar
COPY --from=build /app/references.bin /app/references.bin
COPY --from=build /app/kdtree.bin /app/kdtree.bin

# Drive the real scoring hot path with synthetic requests, then exit. The JVM
# records loaded/linked classes and method profiles into app.aotconf. KDTREE_BIN
# selects the KdTree scorer so its methods (not the IVF fallback) get recorded.
RUN AOT_TRAINING=1 KDTREE_BIN=/app/kdtree.bin REFERENCES_BIN=/app/references.bin \
    java $JVM_FLAGS \
        -XX:AOTMode=record \
        -XX:AOTConfiguration=/app/app.aotconf \
        -jar /app/app.jar; \
    ls -la /app/app.aotconf

# ---- 3. aot-create: assemble the AOT cache ----------------------------------
FROM jdk27 AS aot-create
ARG JVM_FLAGS
WORKDIR /app
COPY --from=build /app/app.jar /app/app.jar
COPY --from=aot-record /app/app.aotconf /app/app.aotconf

RUN java $JVM_FLAGS \
        -XX:AOTMode=create \
        -XX:AOTConfiguration=/app/app.aotconf \
        -XX:AOTCache=/app/app.aot \
        -jar /app/app.jar; \
    ls -la /app/app.aot

# ---- 4. runtime stage -------------------------------------------------------
FROM jdk27 AS runtime
ARG JVM_FLAGS

# jemalloc: the competitor preloads it (MALLOC_ARENA_MAX=1) to curb glibc arena
# fragmentation under the tiny memory budget.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libjemalloc2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home app

WORKDIR /app
COPY --from=build /app/app.jar /app/app.jar
COPY --from=build /app/references.bin /app/references.bin
COPY --from=build /app/kdtree.bin /app/kdtree.bin
COPY --from=aot-create /app/app.aot /app/app.aot

RUN mkdir -p /sockets && chown app:app /sockets

# JVM_FLAGS must match the record/create runs for the AOT cache to load. AOT_FLAGS
# is split out so an A/B run can disable the cache (AOT_FLAGS="") without changing
# the rest. The fd-passing server binds NO TCP port (the Rust LB owns :9999).
# KDTREE_BIN selects the exact-kNN KdTree scorer (default); set SCORER=ivf to fall
# back to the approximate IVF scan over references.bin.
ENV REFERENCES_BIN=/app/references.bin \
    KDTREE_BIN=/app/kdtree.bin \
    FD_SOCKET=/sockets/api.sock \
    READY_FILE=/tmp/rinha-ready \
    NPROBE=6 \
    MAX_NPROBE=16 \
    SCAN_CAP=120000 \
    WORKERS=1 \
    JVM_FLAGS="${JVM_FLAGS}" \
    AOT_FLAGS="-XX:AOTCache=/app/app.aot -Xlog:aot=info" \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 \
    MALLOC_ARENA_MAX=1

USER app

HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=20 \
    CMD test -f "${READY_FILE}" || exit 1

ENTRYPOINT ["sh", "-c", "exec java $JVM_FLAGS $AOT_FLAGS -jar /app/app.jar"]
