#!/usr/bin/env bash
#
# Offline vector-search benchmark for Rinha de Backend 2026.
#
# Replays the labelled corpus (test/test-data.json) straight through the
# production IVF scan in ONE container with the full 1 CPU / 350MB budget — no
# load balancer, no HTTP, no k6. Use it to make the vector search as fast as
# possible in isolation and to confirm a speed change does not regress detection
# accuracy. It reuses the same image as the deployable stack
# (rinha-2026-api:latest): the native binary runs BenchmarkCli when BENCH=1.
#
# Usage:
#   ./scripts/run-bench.sh build           # (re)build the API image (native-image, slow)
#   ./scripts/run-bench.sh run [MODE]      # run a benchmark; MODE = scan|score|full (default scan)
#   ./scripts/run-bench.sh sweep           # run scan|score|full back to back
#   ./scripts/run-bench.sh tsweep [MODE]   # sweep concurrency (1..8) to see CFS throttling per setting
#
# Knobs are passed through as environment variables, e.g.:
#   THREADS=4 NPROBE=12 ./scripts/run-bench.sh run scan
#   BENCH_PASSES=10 BENCH_LIMIT=20000 ./scripts/run-bench.sh run full
#   THREAD_LIST="1 2 4 8" ./scripts/run-bench.sh tsweep scan
#
# `run`/`sweep` do NOT rebuild — run `build` first after changing source.
#
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE="docker compose -f docker-compose.bench.yml"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }

bench_build() {
    log "Building the API image (native-image; this also builds references.bin)..."
    $COMPOSE build
    log "Image built (rinha-2026-cli:latest)."
}

bench_run() {
    local mode="${1:-${BENCH_MODE:-scan}}"
    log "Running vector-search benchmark: mode=${mode} (1 CPU / 350MB)..."
    BENCH_MODE="$mode" $COMPOSE run --rm bench
}

bench_sweep() {
    for m in scan score full; do
        bench_run "$m"
        echo
    done
}

# Sweep concurrency for one mode, keeping the scratch pool (WORKERS) matched to the
# bench worker count (THREADS) so latency reflects scan contention, not pool blocking.
# Each run prints its own "--- cfs throttling ---" block: compare how often a given
# THREADS setting burns the 1-CPU quota inside a 100ms period (the quota wall).
bench_tsweep() {
    local mode="${1:-${BENCH_MODE:-scan}}"
    local list="${THREAD_LIST:-1 2 3 4 6 8}"
    for t in $list; do
        log "concurrency sweep: THREADS=WORKERS=${t}, mode=${mode}"
        THREADS="$t" WORKERS="$t" BENCH_MODE="$mode" $COMPOSE run --rm bench
        echo
    done
}

case "${1:-run}" in
    build) bench_build ;;
    run)   bench_run "${2:-}" ;;
    sweep) bench_sweep ;;
    tsweep) bench_tsweep "${2:-}" ;;
    *)
        echo "usage: $0 {build|run [scan|score|full]|sweep|tsweep [scan|score|full]}" >&2
        exit 2
        ;;
esac
