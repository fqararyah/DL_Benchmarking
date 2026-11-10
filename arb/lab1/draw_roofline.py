import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Draw a roofline plot from the input specifications.")
parser.add_argument("name", nargs="?", help="optional name for the output files (default: timestamp)")
args = parser.parse_args()
if args.name is not None and (not args.name.strip() or args.name in {".", ".."} or "/" in args.name or "\\" in args.name):
    parser.error("name must be a nonempty filename without path separators")

# Bandwidth is in GB/s; performance is in GFLOP/s and intensity in FLOPs/byte.
script_dir = Path(__file__).resolve().parent
with (script_dir / "inp" / "hw_specs.json").open() as config_file:
    hw_specs = json.loads(config_file.read().strip() or "{}")
with (script_dir / "inp" / "kernel_specs.json").open() as config_file:
    kernels = json.loads(config_file.read().strip() or "{}")
if not hw_specs:
    print("No hardware specifications to draw.")
    raise SystemExit(0)
bandwidth = hw_specs["bandwidth_gb_s"]
peak_flops = hw_specs["peak_gflops"]
output_name = args.name if args.name is not None else datetime.now().strftime("%d_%H_%M")
output_dir = script_dir / "out"
output_dir.mkdir(exist_ok=True)
output_file = output_dir / f"roofline_{output_name}.png"

# -------------------------------
# Compute intersection
# -------------------------------
ai_intersect = peak_flops / bandwidth
perf_intersect = peak_flops

# -------------------------------
# Generate Roofline
# -------------------------------
ai = np.logspace(-2, 3, 500)
mem_bound = bandwidth * ai
comp_bound = np.full_like(ai, peak_flops)
roof = np.minimum(mem_bound, comp_bound)

# -------------------------------
# Plot setup
# -------------------------------
plt.figure(figsize=(8, 6))
plt.loglog(ai, roof, label="Roofline", color="black", linewidth=2)
plt.loglog(ai, mem_bound, "--", color="gray", label="Memory BW limit")
plt.loglog(ai, comp_bound, "--", color="red", label="Compute peak")

# Intersection point
plt.scatter(ai_intersect, perf_intersect, color="green", s=100, marker="x", zorder=6)
plt.text(ai_intersect * 0.9, perf_intersect * 1.2,
         f"({ai_intersect:.2f}, {perf_intersect:.1f})",
         fontsize=12, color="#006600", ha="right", va="bottom")

# -------------------------------
# Colors for points
# -------------------------------
tab_colors = list(plt.get_cmap('tab10').colors)
# skip first few (they’re similar to blue/red/green)
avail_colors = tab_colors[3:]  

# -------------------------------
# Measured points and guide lines
# -------------------------------
roofline_efficiency_percent = {}
for i, (label, measurement) in enumerate(kernels.items()):
    oi = measurement["operational_intensity"]
    perf = measurement["measured_gflops"]
    color = avail_colors[i % len(avail_colors)]

    # Compute roofline limit at this OI
    roof_y = min(peak_flops, bandwidth * oi)
    efficiency = perf / roof_y
    roofline_efficiency_percent[label] = efficiency

    # Vertical dashed line to the roofline
    plt.loglog([oi, oi], [1e-2, roof_y], linestyle="--", color=color, alpha=0.6)

    # Scatter point
    plt.scatter(oi, perf, color=color, s=80, zorder=5, label=label)
    plt.annotate(f"{efficiency:.2}", (oi, perf),
                 xytext=(5, 5), textcoords="offset points",
                 color=color, fontsize=9)

# -------------------------------
# Final formatting
# -------------------------------
plt.title("Roofline Model", fontsize=14, weight="bold")
plt.xlabel("Operational Intensity (FLOPs / Byte)", fontsize=12)
plt.ylabel("Performance (GFLOP/s)", fontsize=12)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

# Save image
plt.savefig(output_file, dpi=300)
plt.close()

report_file = output_dir / f"roofline_{output_name}.json"
with report_file.open("w") as report:
    json.dump({"ratio_to_roof": roofline_efficiency_percent}, report, indent=2)
    report.write("\n")

print(f"✅ Roofline plot saved to '{output_file}'")
print(f"📊 Roofline efficiency saved to '{report_file}'")
print(f"🟢 Intersection: OI = {ai_intersect:.2f} FLOPs/Byte, Perf = {perf_intersect:.2f} GFLOP/s")
