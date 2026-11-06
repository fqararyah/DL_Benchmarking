#!/bin/bash
# run.sh — run one kernel ONCE at the current CPU setting and save the result.
#
# Usage: ./run.sh matmul      (compute-heavy app)
#        ./run.sh triad       (memory-heavy app)
#
# First choose the setting with set_governor.sh or set_freq.sh, then run this script.
# The setting is read from the system, so you never type it by hand.
#
# Every run adds one row to data/results.csv:
#   kernel,setting,time_s,avg_power_w,energy_j,temp_start_c,temp_end_c,throttled

cd "$(dirname "$0")"          # so the relative paths below work from any folder

POLICY=/sys/devices/system/cpu/cpufreq/policy0
RESULTS=data/results.csv
TEMP_LIMIT=55                 # C: if the CPU is hotter than this, wait before running

# ---- 1. Which kernel? ---------------------------------------------------------
case "$1" in
  matmul) KERNEL=./kernels/matmul/matmul_exec ;;
  triad)  KERNEL=./kernels/triad/triad ;;
  *)      echo "Usage: $0 matmul|triad"; exit 1 ;;
esac

# ---- 2. Which setting is active? ------------------------------------------------
# Fixed frequency (userspace governor) -> setting is the frequency in MHz, e.g. "1800"
# Any other governor                  -> setting is the governor name, e.g. "powersave"
GOVERNOR=$(cat $POLICY/scaling_governor)
if [[ "$GOVERNOR" == "userspace" ]]; then
  SETTING=$(( $(cat $POLICY/scaling_setspeed) / 1000 ))
else
  SETTING=$GOVERNOR
fi

# ---- 3. Wait until the CPU has cooled down -------------------------------------
get_temp() { vcgencmd measure_temp | tr -dc '0-9.'; }   # "temp=52.1'C" -> 52.1

TEMP_START=$(get_temp)
while (( ${TEMP_START%.*} > TEMP_LIMIT )); do
  echo "CPU at ${TEMP_START} C (limit ${TEMP_LIMIT} C) - cooling down..."
  sleep 5
  TEMP_START=$(get_temp)
done

# ---- 4. Run the kernel and measure time, power and energy ----------------------
echo "=== Running $1 | setting: $SETTING | start temp: ${TEMP_START} C ==="
THROTTLED_BEFORE=$(vcgencmd get_throttled | cut -d= -f2)

OUTPUT=$(./utils/power_wrapper.sh $KERNEL | tee /dev/stderr)   # tee: also show it on screen

TIME_S=$(echo "$OUTPUT"  | grep EXEC_TIME_S | cut -d= -f2)
POWER_W=$(echo "$OUTPUT" | grep AVG_POWER_W | cut -d= -f2)
ENERGY_J=$(echo "$OUTPUT" | grep ENERGY_J   | cut -d= -f2)

# ---- 5. Did the CPU throttle during the run? ------------------------------------
# get_throttled is a bit mask: the lowest 4 bits are "happening now", the upper
# bits are "happened since boot". A change, or any "now" bit, means the run is not valid.
TEMP_END=$(get_temp)
THROTTLED_AFTER=$(vcgencmd get_throttled | cut -d= -f2)
if [[ "$THROTTLED_AFTER" != "$THROTTLED_BEFORE" ]] || (( THROTTLED_AFTER & 0xF )); then
  THROTTLED=yes
else
  THROTTLED=no
fi

# ---- 6. Save and summarise -------------------------------------------------------
mkdir -p data
if [[ ! -f "$RESULTS" ]]; then
  echo "kernel,setting,time_s,avg_power_w,energy_j,temp_start_c,temp_end_c,throttled" > "$RESULTS"
fi
echo "$1,$SETTING,$TIME_S,$POWER_W,$ENERGY_J,$TEMP_START,$TEMP_END,$THROTTLED" >> "$RESULTS"

echo "-------------------------------------------"
echo " Kernel   : $1"
echo " Setting  : $SETTING"
echo " Time     : $TIME_S s"
echo " Power    : $POWER_W W (average)"
echo " Energy   : $ENERGY_J J"
echo " Temp     : $TEMP_START C -> $TEMP_END C"
echo " Throttled: $THROTTLED"
echo " Saved to : $RESULTS"
echo "-------------------------------------------"
if [[ "$THROTTLED" == "yes" ]]; then
  echo "WARNING: the CPU throttled during this run. check_results.py will remove it."
fi
