#!/bin/bash
# power_sampler.sh — sample the Raspberry Pi 5 PMIC and append one line per sample:
#     <unix_time_s> <total_power_W>
# Total power = sum over all PMIC rails of (rail current * rail voltage).
# One vcgencmd + one awk per sample keeps the sampler's own CPU load low
# (~40 samples/s; vcgencmd is the bottleneck).

set -uo pipefail

POWER_OUT="${1:?Usage: power_sampler.sh <output_file>}"

while true; do
  vcgencmd pmic_read_adc | awk -v t="$(date +%s.%N)" '
    # lines look like:  VDD_CORE_A current(7)=1.15197000A   /   VDD_CORE_V volt(15)=0.85958410V
    {
      split($2, kv, "="); val = kv[2] + 0
      rail = substr($1, 1, length($1) - 2)          # strip trailing _A / _V
      if ($2 ~ /^current/) I[rail] = val
      else if ($2 ~ /^volt/) V[rail] = val
    }
    END {
      p = 0
      for (r in I) if (r in V) p += I[r] * V[r]
      if (p > 0) printf "%s %.4f\n", t, p
    }' >> "$POWER_OUT"
done
