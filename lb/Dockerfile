# Build stage: rust:alpine is musl-native, so `cargo build` produces a fully
# static binary without any cross-toolchain setup. Result is ~2-3 MB.
FROM rust:1.83-alpine AS build
RUN apk add --no-cache musl-dev

WORKDIR /lb

# target-cpu=haswell: the evaluation box is a 2014 Mac Mini (i5-4xxxU). Lets
# rustc emit AVX2/BMI2 for any vectorizable byte ops. Free perf since this is a
# Haswell-only (linux-amd64) binary.
ENV RUSTFLAGS="-C target-cpu=haswell"

# Layer-cache the dependency build so source-only edits skip the slow tokio
# compile: build a stub main to resolve+compile deps, then drop the stub
# artifacts before copying real sources. Cargo.lock is committed for
# reproducibility.
COPY Cargo.toml Cargo.lock ./
RUN mkdir src \
    && echo 'fn main() {}' > src/main.rs \
    && cargo build --release \
    && rm -rf src target/release/rinha-lb target/release/rinha-lb.d \
              target/release/deps/rinha_lb-*

COPY src ./src
# CARGO_LTO defaults to the production "fat" LTO. It can be overridden (e.g. to
# "off") only to work around a rustc SIGABRT when cross-building under QEMU
# emulation on a non-amd64 host; the published amd64 image always uses fat LTO.
ARG CARGO_LTO=fat
RUN CARGO_PROFILE_RELEASE_LTO="${CARGO_LTO}" cargo build --release && strip target/release/rinha-lb

# Runtime stage: scratch + a single static binary. No shell, no libc, no PID 1
# init. The HEALTHCHECK uses the binary's own --healthcheck (exec form, no shell).
FROM scratch
COPY --from=build /lb/target/release/rinha-lb /rinha-lb

ENV LB_LISTEN=0.0.0.0:9999 \
    LB_READY_FILE=/lb.ready

HEALTHCHECK --interval=5s --timeout=3s --start-period=30s --retries=20 \
    CMD ["/rinha-lb", "--healthcheck"]

ENTRYPOINT ["/rinha-lb"]
