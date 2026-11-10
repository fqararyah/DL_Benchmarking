#!/bin/bash
# power_wrapper.sh — run a command while sampling PMIC power.
# Prints (machine-friendly, last three lines):
#   EXEC_TIME_S=<wall time of the command>
#   AVG_POWER_W=<mean power over the run>
#   ENERGY_J=<EXEC_TIME_S * AVG_POWER_W>
#
# Usage: ./power_wrapper.sh <command> [args...]

set -uo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
[[ $# -ge 1 ]] || { echo "Usage: $0 <command> [args...]" >&2; exit 1; }

POWER_FILE="$(mktemp)"
"${SCRIPT_DIR}/power_sampler.sh" "$POWER_FILE" &
SAMPLER_PID=$!

stop_sampler() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill "$SAMPLER_PID" 2>/dev/null
    wait "$SAMPLER_PID" 2>/dev/null
  fi
}
trap 'stop_sampler; rm -f "$POWER_FILE"' EXIT

start_ts="$(date +%s.%N)"
"$@"
cmd_rc=$?
end_ts="$(date +%s.%N)"

stop_sampler

elapsed_s="$(awk -v a="$start_ts" -v b="$end_ts" 'BEGIN{printf "%.6f", b-a}')"

# Average of the samples taken while the command was running
avg_w="$(awk -v a="$start_ts" -v b="$end_ts" '
  NF==2 && $1>=a && $1<=b {s+=$2; n++}
  END{ if(n>0) printf "%.6f", s/n; else printf "0.000000" }' "$POWER_FILE")"
n_samples="$(awk -v a="$start_ts" -v b="$end_ts" 'NF==2 && $1>=a && $1<=b {n++} END{print n+0}' "$POWER_FILE")"

energy_j="$(awk -v t="$elapsed_s" -v p="$avg_w" 'BEGIN{printf "%.6f", t*p}')"

[[ "$n_samples" -gt 0 ]] || echo "WARNING: no power samples were collected (is vcgencmd working?)" >&2

echo "POWER_SAMPLES=${n_samples}"
echo "EXEC_TIME_S=${elapsed_s}"
echo "AVG_POWER_W=${avg_w}"
echo "ENERGY_J=${energy_j}"

exit "$cmd_rc"
