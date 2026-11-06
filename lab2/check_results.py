"""
check_results.py - remove disturbed runs from data/results.csv.

Every setting is measured (at least) twice. Background activity can only make a
run slower and more expensive, so for each kernel + setting the run with the
LOWEST energy is taken as the good one. Any other run that is more than
LIMIT % slower or LIMIT % more expensive than it is removed.

Removed rows are moved to data/removed.csv, and the script prints the commands
to measure them again. Run it until it says that all measurements are OK.

Usage:
  python3 check_results.py
"""

import csv
import os

RESULTS_FILE = "data/results.csv"
REMOVED_FILE = "data/removed.csv"
LIMIT = 5.0    # % : normal noise is below ~4%, disturbed runs differ by 10% or more
GOVERNORS = ["performance", "powersave", "ondemand", "schedutil", "conservative"]


def percent_above(value, reference):
    return 100 * (value - reference) / reference


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))     # paths relative to this script
    if not os.path.exists(RESULTS_FILE):
        print(f"No results yet: {RESULTS_FILE} does not exist. Measure first with ./run.sh")
        return

    with open(RESULTS_FILE) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    # Group the runs by kernel + setting
    groups = {}
    for row in rows:
        groups.setdefault((row["kernel"], row["setting"]), []).append(row)

    keep, removed, to_measure = [], [], []
    for (kernel, setting), runs in groups.items():
        measured = [r for r in runs if float(r["energy_j"]) > 0]     # 0 J = power sampling failed
        best = min(measured, key=lambda r: float(r["energy_j"]), default=None)
        for run in runs:
            if run not in measured:
                removed.append(run)
                print(f"REMOVED  {kernel:6} {setting:>11}: no energy measured, repeat the measurement")
                continue
            slower = percent_above(float(run["time_s"]), float(best["time_s"]))
            costlier = percent_above(float(run["energy_j"]), float(best["energy_j"]))
            if run["throttled"] == "yes" or slower > LIMIT or costlier > LIMIT:
                removed.append(run)
                print(f"REMOVED  {kernel:6} {setting:>11}: {slower:+.1f}% time, {costlier:+.1f}% energy"
                      f" (throttled: {run['throttled']})")
            else:
                keep.append(run)
        if len([r for r in runs if r in keep]) < 2:
            to_measure.append((kernel, setting))

    # Write back the good rows, append the removed ones to removed.csv
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(keep)
    if removed:
        new_file = not os.path.exists(REMOVED_FILE)
        with open(REMOVED_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if new_file:
                writer.writeheader()
            writer.writerows(removed)

    if not to_measure:
        print(f"All measurements OK: every setting has 2 runs within {LIMIT}% in time and energy.")
        return

    print("\nMeasure these again (one run each), then run this script again:")
    for kernel, setting in to_measure:
        if setting in GOVERNORS:
            print(f"  sudo ./set_governor.sh {setting} ; ./run.sh {kernel}")
        else:
            print(f"  sudo ./set_freq.sh {setting} ; ./run.sh {kernel}")


if __name__ == "__main__":
    main()
