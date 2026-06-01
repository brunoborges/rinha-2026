#!/usr/bin/env bash
#
# Local test harness for Rinha de Backend 2026 (see docs/EVALUATION.md).
#
# Brings up the deployable stack (nginx + 2 API instances, 1 CPU / 350MB total)
# and runs the OFFICIAL k6 scripts from ./test against it. To stay reliable on
# both Linux and Docker Desktop (Mac), k6 runs as a container attached to the
# stack's bridge network and targets the `nginx` service by name — so the
# upstream test scripts are used unmodified except for the localhost->nginx host
# rewrite (their logic, payloads and scoring are untouched).
#
# Usage:
#   ./scripts/run-tests.sh up       # build + start the stack, wait until healthy
#   ./scripts/run-tests.sh smoke    # up, then run the smoke test (1 VU x 5)
#   ./scripts/run-tests.sh load     # up, then run the full load test (-> 900 rps)
#   ./scripts/run-tests.sh stats    # one-shot `docker stats` of the stack
#   ./scripts/run-tests.sh down     # tear the stack down
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"

COMPOSE="docker compose"
NETWORK="rinha-net"
K6_IMAGE="grafana/k6:latest"
BASE_URL="http://localhost:9999"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err() { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; }

stack_up() {
    log "Building and starting the stack (this also builds references.bin)…"
    $COMPOSE up -d --build --wait --wait-timeout 300
    log "Stack is up. Waiting for ${BASE_URL}/ready via the load balancer…"
    for i in $(seq 1 60); do
        if curl -fsS "${BASE_URL}/ready" >/dev/null 2>&1; then
            log "Load balancer is serving /ready (200)."
            return 0
        fi
        sleep 2
    done
    err "Timed out waiting for ${BASE_URL}/ready"
    $COMPOSE ps
    return 1
}

stack_down() {
    log "Tearing down the stack…"
    $COMPOSE down -v --remove-orphans
}

# Run a k6 script from ./test against the nginx service on the stack network.
# $1 = script filename (test.js | smoke.js)
run_k6() {
    local script="$1"
    local tmp
    tmp="$(mktemp -d)"
    mkdir -p "${tmp}/test"  # test.js writes 'test/results.json' relative to cwd

    # Copy the JS helpers, rewriting the hard-coded localhost target to the
    # in-network service name, plus the dataset (single mount avoids Docker
    # Desktop's "mountpoint outside rootfs" error from nested file binds).
    for f in "$script" k6-summary.js; do
        sed 's#http://localhost:9999#http://nginx:9999#g' "test/$f" > "${tmp}/$f"
    done
    cp "test/test-data.json" "${tmp}/test-data.json"

    log "Running k6 ($script) against the stack…"
    local rc=0
    docker run --rm -i \
        --network "$NETWORK" \
        -e K6_NO_USAGE_REPORT=true \
        -v "${tmp}:/test" \
        -w /test \
        "$K6_IMAGE" run "/test/$script" || rc=$?

    if [ -f "${tmp}/test/results.json" ]; then
        cp "${tmp}/test/results.json" "${ROOT}/test/results.json"
        log "Results written to test/results.json:"
        cat "${ROOT}/test/results.json"
    fi
    rm -rf "$tmp"
    return $rc
}

case "${1:-smoke}" in
    up)    stack_up ;;
    smoke) stack_up && run_k6 smoke.js ;;
    load)  stack_up && run_k6 test.js ;;
    stats) docker stats --no-stream $($COMPOSE ps -q) ;;
    down)  stack_down ;;
    *)     err "unknown command '${1}'"; grep '^#   ' "$0"; exit 2 ;;
esac
