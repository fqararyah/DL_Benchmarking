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
TEMP_LIMIT=55                 # C: a run only starts when the CPU is at or below this
PASS_TIME=30                  # s: length of one cooling pass (temperature checked every 5 s)
PASSES=2                      # number of cooling passes before giving up

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
# Up to PASSES passes of PASS_TIME seconds. The run starts as soon as the CPU is
# cool enough. If it is still too hot after the last pass, something keeps the CPU
# warm (programs in the background, GUI, fan): stop and tell the user.
get_temp() { vcgencmd measure_temp | tr -dc '0-9.'; }   # "temp=52.1'C" -> 52.1
too_hot()  { (( $(get_temp | cut -d. -f1) > TEMP_LIMIT )); }

for (( pass = 1; pass <= PASSES; pass++ )); do
  waited=0
  while too_hot && (( waited < PASS_TIME )); do
    echo "CPU at $(get_temp) C (limit ${TEMP_LIMIT} C) - cooling down: pass $pass of $PASSES, ${waited}/${PASS_TIME} s"
    sleep 5
    waited=$(( waited + 5 ))
  done
  too_hot || break
done

if too_hot; then
  echo "ERROR: the CPU is still at $(get_temp) C after $(( PASSES * PASS_TIME )) s (limit ${TEMP_LIMIT} C)."
  echo "Nothing was run or saved. Close all other programs (VS Code, browser, ...),"
  echo "make sure the GUI is off, check the fan with ./status.sh, or reboot."
  echo "Then run the same command again."
  exit 1
fi
TEMP_START=$(get_temp)

# ---- 4. Run the kernel and measure time, power and energy ----------------------
# Run number = runs of this kernel + setting already in results.csv, plus this one
PREVIOUS_RUNS=0
if [[ -f "$RESULTS" ]]; then
  PREVIOUS_RUNS=$(grep -c "^$1,$SETTING," "$RESULTS")
fi
RUN_NUMBER=$(( PREVIOUS_RUNS + 1 ))

echo "=== Running $1 (run $RUN_NUMBER for this setting) | setting: $SETTING | start temp: ${TEMP_START} C ==="
THROTTLED_BEFORE=$(vcgencmd get_throttled | cut -d= -f2)

OUTPUT=$(./utils/power_wrapper.sh $KERNEL)
echo "$OUTPUT"                # show the kernel output and the measurements

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
echo " Run      : $RUN_NUMBER for this setting (every setting needs 2 good runs)"
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
