#!/bin/bash
# status.sh — show the current CPU frequency setting, temperature and throttling state.
# Use it whenever you are unsure what the Raspberry Pi is currently set to.
# Usage: ./status.sh

POLICY=/sys/devices/system/cpu/cpufreq/policy0   # one policy controls all 4 cores

GOVERNOR=$(cat $POLICY/scaling_governor)
CUR_MHZ=$(( $(cat $POLICY/scaling_cur_freq) / 1000 ))
MIN_MHZ=$(( $(cat $POLICY/scaling_min_freq) / 1000 ))
MAX_MHZ=$(( $(cat $POLICY/scaling_max_freq) / 1000 ))
TEMP=$(vcgencmd measure_temp | tr -dc '0-9.')          # "temp=52.1'C" -> 52.1
THROTTLED=$(vcgencmd get_throttled | cut -d= -f2)      # "throttled=0x0" -> 0x0

echo "-------------------------------------------"
echo " Governor      : $GOVERNOR"
echo " Frequency now : $CUR_MHZ MHz"
echo " Allowed range : $MIN_MHZ - $MAX_MHZ MHz"
echo " Temperature   : $TEMP C"
echo " Throttled     : $THROTTLED   (0x0 = never throttled since boot)"
echo "-------------------------------------------"
