#!/bin/bash
# set_freq.sh — lock all 4 cores to one fixed frequency (uses the "userspace" governor).
# Usage: sudo ./set_freq.sh <MHz>
#   e.g. sudo ./set_freq.sh 1800

POLICY=/sys/devices/system/cpu/cpufreq/policy0
MHZ="$1"
AVAILABLE_KHZ=$(cat $POLICY/scaling_available_frequencies)
AVAILABLE_MHZ=$(for f in $AVAILABLE_KHZ; do echo -n "$((f / 1000)) "; done)

if [[ $EUID -ne 0 ]]; then
  echo "Error: this script must be run with sudo:  sudo $0 $MHZ"
  exit 1
fi
if [[ -z "$MHZ" ]] || ! grep -qw "$MHZ" <<< "$AVAILABLE_MHZ"; then
  echo "Usage: sudo $0 <MHz>"
  echo "Available frequencies (MHz): $AVAILABLE_MHZ"
  exit 1
fi

# Allow the full hardware range, switch to userspace, then request the frequency (in kHz)
cat $POLICY/cpuinfo_max_freq > $POLICY/scaling_max_freq
cat $POLICY/cpuinfo_min_freq > $POLICY/scaling_min_freq
echo userspace > $POLICY/scaling_governor
echo $((MHZ * 1000)) > $POLICY/scaling_setspeed
sleep 0.5

echo "Frequency locked to: $MHZ MHz"
"$(dirname "$0")/status.sh"
