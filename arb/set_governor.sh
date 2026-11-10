#!/bin/bash
# set_governor.sh — select a CPU governor for all 4 cores.
# Usage: sudo ./set_governor.sh <governor>
#   e.g. sudo ./set_governor.sh performance
#        sudo ./set_governor.sh powersave

POLICY=/sys/devices/system/cpu/cpufreq/policy0
GOVERNOR="$1"
AVAILABLE=$(cat $POLICY/scaling_available_governors)

if [[ $EUID -ne 0 ]]; then
  echo "Error: this script must be run with sudo:  sudo $0 $GOVERNOR"
  exit 1
fi
if [[ -z "$GOVERNOR" ]] || ! grep -qw "$GOVERNOR" <<< "$AVAILABLE"; then
  echo "Usage: sudo $0 <governor>"
  echo "Available governors: $AVAILABLE"
  exit 1
fi

# Allow the full hardware frequency range, then select the governor
cat $POLICY/cpuinfo_max_freq > $POLICY/scaling_max_freq
cat $POLICY/cpuinfo_min_freq > $POLICY/scaling_min_freq
echo "$GOVERNOR" > $POLICY/scaling_governor
sleep 0.5

echo "Governor set to: $GOVERNOR"
"$(dirname "$0")/status.sh"
