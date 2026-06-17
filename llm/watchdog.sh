#!/bin/bash
# llama-watchdog — VRAM guardian + stall detector for murderbot
#
# Restarts the llama-server Docker container if:
#   1. Free VRAM drops below VRAM_MIN_MIB (protects Jellyfin NVENC transcoding)
#   2. A slot has been occupied with no new tokens for STALL_TIMEOUT_S
#      (catches the PR #22907 cascade and other hung-inference states)
#
# Runs as a systemd service. See llama-watchdog.service.
# Install via: ~/claude/manual_runs/install-llama-watchdog.sh

CONTAINER=llama-server
VRAM_MIN_MIB=800          # Restart if free VRAM drops below this
STALL_TIMEOUT_S=300       # Restart if no new tokens for 5 min while slot busy
CHECK_INTERVAL_S=30       # Poll interval
COOLDOWN_S=180            # Minimum seconds between restarts
METRICS_URL=http://127.0.0.1:8088/metrics

LAST_RESTART=0
LAST_TOKENS_TOTAL=""
LAST_TOKENS_TIME=0

log() {
    echo "$(date -Iseconds) llama-watchdog: $*"
    logger -t llama-watchdog "$*" 2>/dev/null || true
}

get_metric() {
    local name="$1"
    curl -sf --max-time 5 "$METRICS_URL" 2>/dev/null \
        | awk -v n="^${name} " '$0 ~ n {print $2; exit}'
}

restart_container() {
    local reason="$1"
    local now; now=$(date +%s)
    if (( now - LAST_RESTART < COOLDOWN_S )); then
        log "SKIP restart (${COOLDOWN_S}s cooldown): $reason"
        return 1
    fi
    log "RESTART: $reason"
    docker restart "$CONTAINER" 2>&1 | while IFS= read -r line; do log "docker: $line"; done
    LAST_RESTART=$(date +%s)
    LAST_TOKENS_TOTAL=""
    LAST_TOKENS_TIME=0
    return 0
}

log "starting (container=$CONTAINER, vram_min=${VRAM_MIN_MIB}MiB, stall=${STALL_TIMEOUT_S}s)"

while true; do
    sleep "$CHECK_INTERVAL_S"

    # ── 1. VRAM protection ───────────────────────────────────────────────────
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
               | head -1 | tr -d ' ')
    if [[ "$free_mib" =~ ^[0-9]+$ && "$free_mib" -lt "$VRAM_MIN_MIB" ]]; then
        restart_container "VRAM critically low: ${free_mib} MiB free (min: ${VRAM_MIN_MIB} MiB)"
        continue
    fi

    # ── 2. Stall detection ───────────────────────────────────────────────────
    requests=$(get_metric "llamacpp:requests_processing")
    tokens=$(get_metric "llamacpp:tokens_predicted_total")
    now=$(date +%s)

    if [[ "$requests" =~ ^[0-9]+$ && "$requests" -gt 0 && -n "$tokens" ]]; then
        if [[ "$tokens" != "$LAST_TOKENS_TOTAL" ]]; then
            # Progress: tokens are being generated
            LAST_TOKENS_TOTAL="$tokens"
            LAST_TOKENS_TIME="$now"
        elif [[ "$LAST_TOKENS_TIME" -gt 0 ]] && (( now - LAST_TOKENS_TIME > STALL_TIMEOUT_S )); then
            stall_s=$(( now - LAST_TOKENS_TIME ))
            restart_container "inference stall: slot occupied, no new tokens for ${stall_s}s"
        fi
    else
        # No active slot — reset stall timer
        LAST_TOKENS_TIME=0
    fi
done
