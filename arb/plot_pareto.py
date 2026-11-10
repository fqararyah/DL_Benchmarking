"""
plot_pareto.py - Pareto analysis of the results of one kernel.

Reads data/results.csv (written by run.sh) and:
  1. averages repeated runs of the same setting,
  2. prints a table with time, power, energy and the change vs. Performance,
  3. marks the Pareto-optimal settings,
  4. saves two plots side by side in plots/pareto_<kernel>.png:
       left : energy (J) vs. execution time (s)
       right: average power (W) vs. execution time (s)
     In both plots the best corner is bottom-left (fast and cheap).

Pareto rule: setting A DOMINATES setting B if A is no worse than B in both time
and cost (energy or power), and strictly better in at least one of them.
A setting is Pareto-optimal (on the front) if no other setting dominates it.

Usage:
  python3 plot_pareto.py matmul
  python3 plot_pareto.py triad
"""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")                # save to a file, no window needed
import matplotlib.pyplot as plt

RESULTS_FILE = "data/results.csv"
PLOTS_DIR = "plots"
GOVERNORS = ["performance", "powersave", "ondemand", "schedutil", "conservative"]

FRONT_COLOR = "#2a78d6"      # blue: Pareto-optimal settings
OTHER_COLOR = "#a3a29b"      # gray: dominated settings


def load_results(kernel):
    """Return one dict per setting, with time/power/energy averaged over repeated runs."""
    runs_per_setting = {}
    with open(RESULTS_FILE) as f:
        for row in csv.DictReader(f):
            if row["kernel"] == kernel and row["throttled"] == "no":
                runs_per_setting.setdefault(row["setting"], []).append(row)

    results = []
    for setting, runs in runs_per_setting.items():
        n = len(runs)
        results.append({
            "setting": setting,
            "runs": n,
            "time": sum(float(r["time_s"]) for r in runs) / n,
            "power": sum(float(r["avg_power_w"]) for r in runs) / n,
            "energy": sum(float(r["energy_j"]) for r in runs) / n,
        })
    results.sort(key=lambda r: r["time"])
    return results


def dominates(a, b, cost):
    """True if a is no worse than b in time AND cost, and strictly better in at least one."""
    no_worse = a["time"] <= b["time"] and a[cost] <= b[cost]
    better = a["time"] < b["time"] or a[cost] < b[cost]
    return no_worse and better


def on_front(point, results, cost):
    """True if no other setting dominates this one."""
    return not any(dominates(other, point, cost) for other in results)


def find_reference(results):
    """Performance mode = the 'performance' governor, or 2400 MHz if it was not measured."""
    for name in ["performance", "2400"]:
        for r in results:
            if r["setting"] == name:
                return r
    return None


def print_table(kernel, results, reference):
    print(f"\nResults for {kernel} (averaged over repeated runs, reference = {reference['setting']})")
    print(f"{'setting':>12} {'runs':>4} {'time[s]':>8} {'power[W]':>9} {'energy[J]':>10}"
          f" {'time vs ref':>12} {'energy vs ref':>14} {'front(E)':>9} {'front(P)':>9}")
    for r in results:
        time_change = 100 * (r["time"] / reference["time"] - 1)
        energy_change = 100 * (r["energy"] / reference["energy"] - 1)
        front_e = "yes" if on_front(r, results, "energy") else "-"
        front_p = "yes" if on_front(r, results, "power") else "-"
        print(f"{r['setting']:>12} {r['runs']:>4} {r['time']:>8.2f} {r['power']:>9.2f} {r['energy']:>10.1f}"
              f" {time_change:>11.1f}% {energy_change:>13.1f}% {front_e:>9} {front_p:>9}")
    print()


def draw(ax, results, cost, ylabel):
    front = [r for r in results if on_front(r, results, cost)]

    # Pareto front as a line (results are sorted by time), then all points on top
    ax.plot([r["time"] for r in front], [r[cost] for r in front],
            color=FRONT_COLOR, linewidth=2, zorder=1, label="Pareto front")
    for r in results:
        color = FRONT_COLOR if r in front else OTHER_COLOR
        marker = "s" if r["setting"] in GOVERNORS else "o"     # square = governor, circle = fixed MHz
        ax.scatter(r["time"], r[cost], color=color, marker=marker, s=50,
                   edgecolors="white", linewidths=1.5, zorder=2)
        ax.annotate(r["setting"], (r["time"], r[cost]), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#333333")

    ax.margins(x=0.15, y=0.15)          # room for the text labels
    ax.set_xlabel("Execution time (s)   (lower is better)")
    ax.set_ylabel(ylabel + "   (lower is better)")
    ax.grid(True, color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ["matmul", "triad"]:
        print("Usage: python3 plot_pareto.py matmul|triad")
        return
    kernel = sys.argv[1]

    os.chdir(os.path.dirname(os.path.abspath(__file__)))     # paths relative to this script
    if not os.path.exists(RESULTS_FILE):
        print(f"No results yet: {RESULTS_FILE} does not exist. Measure first with ./run.sh")
        return
    results = load_results(kernel)
    if len(results) < 2:
        print(f"Need at least 2 settings for {kernel} in {RESULTS_FILE}, found {len(results)}.")
        return

    reference = find_reference(results)
    if reference is None:
        print("Measure the Performance mode first (performance governor or 2400 MHz).")
        return
    print_table(kernel, results, reference)

    fig, (ax_energy, ax_power) = plt.subplots(1, 2, figsize=(12, 5))
    draw(ax_energy, results, "energy", "Energy (J)")
    draw(ax_power, results, "power", "Average power (W)")
    ax_energy.set_title("Energy vs. time")
    ax_power.set_title("Average power vs. time")
    ax_energy.legend(loc="best", fontsize=8)
    fig.suptitle(f"{kernel}: square = governor, circle = fixed frequency (MHz), blue = Pareto front")
    fig.tight_layout()

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out_file = f"{PLOTS_DIR}/pareto_{kernel}.png"
    fig.savefig(out_file, dpi=150)
    print(f"Plot saved to {out_file}")


if __name__ == "__main__":
    main()
