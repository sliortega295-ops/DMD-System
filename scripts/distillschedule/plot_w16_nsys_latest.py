#!/usr/bin/env python3
"""Plot the latest W16 formal timing and Nsight diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    "sequential": "#6B7280",
    "pipeline": "#0072B2",
    "compute": "#009E73",
    "nccl": "#E69F00",
    "idle": "#D7DCE2",
    "ink": "#17202A",
    "muted": "#5D6875",
    "grid": "#DCE3EA",
    "accent": "#CC79A7",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax, letter: str) -> None:
    ax.text(-0.09, 1.08, letter, transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="top", color=COLORS["ink"])


def tidy(ax) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def assert_partition(devices: list[dict], label: str) -> None:
    for device in devices:
        total = (device["compute_union_fraction"] +
                 device["nccl_union_fraction"] +
                 device["idle_fraction"])
        if abs(total - 1.0) > 2e-6:
            raise ValueError(f"{label} GPU{device['device_id']} is not a partition: {total}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--basename", default="iteration_205_w16_nsys_latest")
    args = parser.parse_args()
    root = args.root.resolve()
    manifests = root / "artifacts" / "manifests"
    figures = root / "artifacts" / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    cm_path = manifests / "iteration_205_w16_critic_matched_result.json"
    balance_path = manifests / "iteration_205_w16_gc_balance_result.json"
    raw_path = manifests / "iteration_205_w16_nsys_raw_analysis.json"
    result_path = manifests / "iteration_205_w16_nsys_result.json"
    full5_path = manifests / "iteration_205_w16_full5_result.json"
    cm, balance, raw, result, full5 = map(load, (cm_path, balance_path, raw_path, result_path, full5_path))

    for label in ("w4_generator", "w4_critic", "w16_generator", "w16_critic"):
        assert_partition(raw["reports"][label]["devices"], label)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    ax_a, ax_b, ax_c, ax_d = axes.flat

    categories = ["Role-local\n8G node + 8C node", "Pair-local\n4G+4C per node"]
    seq = [cm["formal_runs"]["role_local_sequential"]["makespan_seconds"],
           cm["formal_runs"]["pair_local_sequential"]["makespan_seconds"]]
    pipe = [cm["formal_runs"]["role_local_pipeline"]["makespan_seconds"],
            cm["formal_runs"]["pair_local_pipeline"]["makespan_seconds"]]
    x = np.arange(len(categories)); width = 0.34
    b1 = ax_a.bar(x - width/2, seq, width, label="Sequential", color=COLORS["sequential"])
    b2 = ax_a.bar(x + width/2, pipe, width, label="Pipeline", color=COLORS["pipeline"])
    for bars in (b1, b2): ax_a.bar_label(bars, fmt="%.1f s", padding=3, fontsize=9)
    for i, speedup in enumerate((cm["comparisons"]["role_local_pipeline_speedup"], cm["comparisons"]["pair_local_pipeline_speedup"])):
        ax_a.text(i, max(seq[i], pipe[i]) + 16, f"{speedup:.3f}×", ha="center", color=COLORS["accent"], fontweight="bold", fontsize=11)
    ax_a.set_ylim(0, 145); ax_a.set_xticks(x, categories); ax_a.set_ylabel("Formal makespan (s)")
    ax_a.set_title("Fixed work: 32 samples, 4 Critic updates, W4"); ax_a.legend(frameon=False, loc="upper right")
    tidy(ax_a); panel_label(ax_a, "A")

    waves = np.array([4, 8, 16])
    throughput = np.array([balance["wave_sweep"][str(w)]["samples_per_second"] for w in waves])
    makespan = np.array([balance["wave_sweep"][str(w)]["makespan_seconds"] for w in waves])
    ax_b.plot(waves, throughput, marker="o", linewidth=2.4, markersize=7, color=COLORS["pipeline"])
    for w, t, m in zip(waves, throughput, makespan):
        ax_b.annotate(f"{t:.3f} samples/s\n({m:.1f} s)", (w, t), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    gain = balance["steady_state_fit"]["throughput_gain_w16_over_w4"]
    ax_b.text(0.04, 0.92, f"W16 vs W4: +{gain*100:.1f}% throughput", transform=ax_b.transAxes, ha="left", va="top", fontweight="bold", color=COLORS["accent"], bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7ECF4", edgecolor="#D8A9C8"))
    ax_b.set_xlim(2.5, 17.5); ax_b.set_ylim(0.40, 0.53); ax_b.set_xticks(waves)
    ax_b.set_xlabel("Pipeline waves"); ax_b.set_ylabel("Formal throughput (samples/s)")
    ax_b.set_title("Longer runs amortize pipeline fill/drain"); tidy(ax_b); panel_label(ax_b, "B")

    labels = ["W4\nGenerator", "W4\nCritic", "W16\nGenerator", "W16\nCritic"]
    keys = ["w4_generator", "w4_critic", "w16_generator", "w16_critic"]
    compute = np.array([raw["reports"][k]["compute_union_fraction_summary"]["mean"] for k in keys]) * 100
    nccl = np.array([raw["reports"][k]["nccl_union_fraction_summary"]["mean"] for k in keys]) * 100
    idle = 100 - compute - nccl; pos = np.arange(len(labels))
    ax_c.bar(pos, compute, color=COLORS["compute"], label="Compute kernels")
    ax_c.bar(pos, nccl, bottom=compute, color=COLORS["nccl"], hatch="///", label="NCCL kernels")
    ax_c.bar(pos, idle, bottom=compute+nccl, color=COLORS["idle"], label="No kernel active")
    for i in range(len(labels)):
        ax_c.text(i, compute[i]/2, f"{compute[i]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        if nccl[i] > 3: ax_c.text(i, compute[i]+nccl[i]/2, f"{nccl[i]:.1f}%", ha="center", va="center", color=COLORS["ink"], fontweight="bold", fontsize=8)
    ax_c.set_xticks(pos, labels); ax_c.set_ylim(0, 100); ax_c.set_ylabel("Mean fraction of capture (%)")
    ax_c.set_title("Nsight activity: Critic communication remains exposed")
    ax_c.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02)); tidy(ax_c); panel_label(ax_c, "C")

    gen = sorted(raw["reports"]["w16_generator"]["devices"], key=lambda d: d["device_id"])
    critic = sorted(raw["reports"]["w16_critic"]["devices"], key=lambda d: d["device_id"])
    devices = gen + critic; names = [f"G{i}" for i in range(8)] + [f"C{i}" for i in range(8)]
    d_compute = np.array([d["compute_union_fraction"] for d in devices]) * 100
    d_nccl = np.array([d["nccl_union_fraction"] for d in devices]) * 100
    d_idle = 100 - d_compute - d_nccl; dx = np.arange(16)
    ax_d.bar(dx, d_compute, color=COLORS["compute"]); ax_d.bar(dx, d_nccl, bottom=d_compute, color=COLORS["nccl"], hatch="///"); ax_d.bar(dx, d_idle, bottom=d_compute+d_nccl, color=COLORS["idle"])
    ax_d.axvline(7.5, color=COLORS["muted"], linewidth=1.2, linestyle="--")
    gen_ratio = result["derived_findings"]["w16_generator_compute_imbalance_ratio_max_over_min"]
    ceiling = result["derived_findings"]["optimistic_idle_elimination_ceiling"]["maximum_speedup"]
    critic_nccl = result["derived_findings"]["w16_critic_mean_nccl_fraction"]
    ax_d.text(0.02, 0.96, f"Generator compute spread: {gen_ratio:.3f}×", transform=ax_d.transAxes, va="top", color=COLORS["accent"], fontweight="bold")
    ax_d.text(0.98, 0.96, f"Critic NCCL mean: {critic_nccl*100:.1f}%", transform=ax_d.transAxes, va="top", ha="right", color=COLORS["accent"], fontweight="bold")
    ax_d.text(0.5, 0.04, f"Optimistic idle-elimination ceiling ≤ {ceiling:.3f}× (not expected speedup)", transform=ax_d.transAxes, ha="center", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAB4BE", alpha=0.95))
    ax_d.set_xticks(dx, names); ax_d.set_ylim(0, 100); ax_d.set_ylabel("Fraction of capture (%)")
    ax_d.set_title("W16 per-GPU activity: Generator imbalance vs Critic AllReduce"); tidy(ax_d); panel_label(ax_d, "D")

    fig.suptitle("Helios / DistillSchedule — latest W16 scheduling and Nsight profile", fontsize=19, fontweight="bold", color=COLORS["ink"], y=0.975)
    fig.text(0.5, 0.935, "16 × A100 40GB | role-local 8G+8C | formal timing first, profiler diagnostics second", ha="center", fontsize=10.5, color=COLORS["muted"])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.115, top=0.86, hspace=0.38, wspace=0.27)
    fig.text(0.01, 0.025, "Claim boundary: A–B are unprofiled formal systems timings. C–D are diagnostic Nsight interval unions; no quality/convergence claim. Continuous Pipeline Full5 remains Not Run.", fontsize=8.5, color=COLORS["muted"])

    png = figures / f"{args.basename}.png"; svg = figures / f"{args.basename}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight"); fig.savefig(svg, bbox_inches="tight"); plt.close(fig)

    manifest = {
        "schema_version": "distillschedule.w16_nsys_figure.v1", "status": "PASS",
        "title": "Helios / DistillSchedule — latest W16 scheduling and Nsight profile",
        "inputs": [{"path": str(p.relative_to(root)), "sha256": sha256(p)} for p in (cm_path, balance_path, raw_path, result_path, full5_path)],
        "outputs": [{"path": str(png.relative_to(root)), "bytes": png.stat().st_size, "sha256": sha256(png)}, {"path": str(svg.relative_to(root)), "bytes": svg.stat().st_size, "sha256": sha256(svg)}],
        "panels": {"A": "formal fixed-work sequential versus pipeline", "B": "formal pipeline wave-throughput sweep", "C": "mean Nsight activity partitions", "D": "per-GPU W16 Nsight activity partitions"},
        "claim_boundary": "Formal speed claims use unprofiled data. Nsight panels are diagnostic. Pipeline Full5 is Not Run.",
    }
    manifest_path = manifests / f"{args.basename}_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "png": str(png), "svg": str(svg), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
