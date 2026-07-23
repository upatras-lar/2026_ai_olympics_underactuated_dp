"""
Analyze saved Pendubot swing-up experiment data.

Computes, per experiment and averaged across experiments:
  - Uptime:  U = (1/N_ep) * sum_k  I[x_k in X_up]
             i.e. fraction of logged timesteps the pendubot spent upright.
  - Success: whether the controller swung up and held upright for at
             least HOLD_DURATION seconds (non-disturbance), or additionally
             recovered and re-held upright after every disturbance
             (disturbance experiments).

Assumes two pickle files were produced by the modified data-collection
script: one for non-disturbance runs, one for disturbance runs, each
containing the dict:
    {
        "all_timestamps": [...],   # list of N_EXPERIMENTS lists of floats (s)
        "all_frequencies": [...],
        "all_data": [...],         # list of N_EXPERIMENTS lists of
                                    #   (config, vel, torque) tuples
        "urls": [...],
        "dt": float,
        "Tf": float,
        "N_EXPERIMENTS": int,
    }

NOTE ON DISTURBANCES: the data-collection script as written does not log
*when* disturbances were injected. Recovering per-disturbance success
requires that information. Two ways to supply it:
  1. If you add a "disturbance_times" entry to the saved dict (list of
     per-experiment lists of times, in seconds since experiment start),
     this script will pick it up automatically.
  2. Otherwise, fill in DISTURBANCE_TIMES below by hand, indexed the same
     way as the experiments in the disturbance pickle.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import ssqpy

# ---------------------------------------------------------------------------
# Configuration - adjust to match your setup
# ---------------------------------------------------------------------------

NON_DISTURBANCE_PKL = "data/swingup_results.pkl"
DISTURBANCE_PKL = "data-disturbance/swingup_results.pkl"

TIP_HEIGHT_THRESHOLD = (
    0.09  # matches the "Minimum tip height" line in the plots
)
HOLD_DURATION = (
    5.0  # seconds required continuously upright to count as success
)

# Manual fallback if disturbance times aren't stored in the pickle.
# Outer list: one entry per experiment, in the same order as all_data.
# Inner list: disturbance injection times (s, relative to experiment start).
DISTURBANCE_TIMES = [[20.0, 40.0]] * 10

# ---------------------------------------------------------------------------
# Model (needed only to compute tip height via forward kinematics)
# ---------------------------------------------------------------------------

ssqpy.setSilentMode()

model = ssqpy.model.Model(
    1,  # horizon doesn't matter here, we only use forwardKinematics
    0.01,
    urdf_path="pendubot.urdf",
    actuated_joints=[0],
    solver_mode=ssqpy.model.SolverMode.InverseDynamics,
)


def load_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_upright_mask(exp_data, tip_threshold=TIP_HEIGHT_THRESHOLD):
    """
    exp_data: list of (config, vel, torque) tuples for one experiment.
    Returns (upright_mask, tip_positions) where upright_mask[k] is True iff
    x_k is considered in X_up.
    """
    configs = np.array([d[0] for d in exp_data])
    tip_positions = np.array(
        [model.forwardKinematics(cfg, "tip")[0][2] for cfg in configs]
    )
    upright_mask = tip_positions > tip_threshold
    return upright_mask, tip_positions


def compute_uptime(upright_mask):
    """U = (1/N_ep) * sum(indicator) -> fraction of timesteps upright."""
    if len(upright_mask) == 0:
        return 0.0
    return float(np.mean(upright_mask))


def longest_upright_hold(timestamps, upright_mask):
    """
    Longest contiguous duration (s) spent inside X_up, using the actual
    (non-uniform, since the loop isn't hard real-time) timestamps.
    """
    timestamps = np.asarray(timestamps)
    best = 0.0
    run_start = None
    for i, up in enumerate(upright_mask):
        if up:
            if run_start is None:
                run_start = timestamps[i]
            best = max(best, timestamps[i] - run_start)
        else:
            run_start = None
    return best


def evaluate_non_disturbance_experiment(exp_data, exp_timestamps):
    upright_mask, _ = compute_upright_mask(exp_data)
    uptime = compute_uptime(upright_mask)
    hold = longest_upright_hold(exp_timestamps, upright_mask)
    return {
        "uptime": uptime,
        "longest_hold_s": hold,
        "success": hold >= HOLD_DURATION,
    }


def evaluate_disturbance_experiment(
    exp_data, exp_timestamps, disturbance_times
):
    """
    Splits the episode into segments: [start, d1), [d1, d2), ..., [dN, end].
    Success requires that EVERY segment contains a contiguous upright hold
    of at least HOLD_DURATION seconds (i.e. the controller both reached
    upright initially and recovered after each disturbance).
    """
    upright_mask, _ = compute_upright_mask(exp_data)
    timestamps = np.asarray(exp_timestamps)
    uptime = compute_uptime(upright_mask)

    if len(timestamps) == 0:
        return {"uptime": 0.0, "per_segment_success": [], "success": False}

    segment_bounds = [0.0] + list(disturbance_times) + [timestamps[-1] + 1e-9]

    per_segment_success = []
    for i in range(len(segment_bounds) - 1):
        t_start, t_end = segment_bounds[i], segment_bounds[i + 1]
        seg_idx = np.where((timestamps >= t_start) & (timestamps < t_end))[0]
        if len(seg_idx) == 0:
            per_segment_success.append(False)
            continue
        hold = longest_upright_hold(timestamps[seg_idx], upright_mask[seg_idx])
        per_segment_success.append(hold >= HOLD_DURATION)

    return {
        "uptime": uptime,
        "per_segment_success": per_segment_success,
        "success": any(per_segment_success),
    }


def analyze_file(path, disturbance=False, disturbance_times_all=None):
    results = load_results(path)
    all_timestamps = results["all_timestamps"]
    all_data = results["all_data"]
    n_exp = len(all_data)

    # Prefer disturbance times saved in the pickle itself, if present.
    if disturbance:
        stored_times = results.get("disturbance_times", None)
        times_source = (
            stored_times if stored_times is not None else disturbance_times_all
        )
        if times_source is None or len(times_source) != n_exp:
            raise ValueError(
                f"Need disturbance_times for all {n_exp} experiments in '{path}'. "
                f"Add them to the pickle under 'disturbance_times' or fill in "
                f"DISTURBANCE_TIMES in this script."
            )

    per_experiment = []
    for i in range(n_exp):
        if disturbance:
            res = evaluate_disturbance_experiment(
                all_data[i], all_timestamps[i], times_source[i]
            )
        else:
            res = evaluate_non_disturbance_experiment(
                all_data[i], all_timestamps[i]
            )
        per_experiment.append(res)

    successes = [r["success"] for r in per_experiment]
    uptimes = [r["uptime"] for r in per_experiment]

    summary = {
        "per_experiment": per_experiment,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_uptime": float(np.mean(uptimes)) if uptimes else 0.0,
    }
    return summary


def compute_stats(values):
    """Mean, median, std, and 25th/75th percentiles for a list of values."""
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
    }


def plot_summary(nd_summary, d_summary, save_path="summary.pdf"):
    """
    Overall summary plot across both experiment sets:
      - Left panel:  uptime distribution per group (filled box = IQR,
        black line = median, whiskers = min/max, diamond = mean, dots =
        individual experiments), with mean/std and p25/p75 annotated.
      - Right panel: success rate per group, as a percentage.
    Returns the computed per-group uptime stats dict.
    """
    groups = ["Non-disturbance", "Disturbance"]
    colors = ["#4C72B0", "#DD8452"]  # muted blue / muted orange
    light_colors = ["#A9C4E4", "#F2C29B"]  # lighter fills for boxes

    uptimes = [
        [r["uptime"] for r in nd_summary["per_experiment"]],
        [r["uptime"] for r in d_summary["per_experiment"]],
    ]
    success_rates = [nd_summary["success_rate"], d_summary["success_rate"]]
    uptime_stats = {g: compute_stats(u) for g, u in zip(groups, uptimes)}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "text.color": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6), facecolor="white")
    fig.suptitle(
        "Pendubot Swing-Up Performance Summary", fontsize=15, fontweight="bold"
    )

    # --------------------------------------------------------------------
    # Left panel: uptime distribution
    # --------------------------------------------------------------------
    ax = axes[0]
    ax.set_facecolor("#FAFAFA")
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    bp = ax.boxplot(
        uptimes,
        labels=groups,
        showmeans=True,
        patch_artist=True,
        widths=0.5,
        whis=[0, 100],  # whiskers span min/max rather than 1.5*IQR
        meanprops=dict(
            marker="D",
            markerfacecolor="white",
            markeredgecolor="#333333",
            markersize=8,
            markeredgewidth=1.4,
            zorder=5,
        ),
        medianprops=dict(color="#222222", linewidth=2.2, zorder=4),
        whiskerprops=dict(color="#555555", linewidth=1.3),
        capprops=dict(color="#555555", linewidth=1.3),
        boxprops=dict(linewidth=1.3),
        zorder=3,
    )
    for patch, fc, ec in zip(bp["boxes"], light_colors, colors):
        patch.set_facecolor(fc)
        patch.set_edgecolor(ec)
        patch.set_alpha(0.85)

    # Jittered scatter of individual experiment values on top of each box
    rng = np.random.default_rng(0)
    for i, (u, c) in enumerate(zip(uptimes, colors)):
        jitter = rng.uniform(-0.09, 0.09, size=len(u))
        ax.scatter(
            np.full(len(u), i + 1) + jitter,
            u,
            color=c,
            edgecolor="white",
            linewidth=0.6,
            s=45,
            alpha=0.85,
            zorder=6,
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.set_ylabel("Uptime", fontsize=12)
    ax.set_title(
        "Uptime distribution across experiments", fontsize=12.5, pad=12
    )
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#222222", lw=2.2, label="Median"),
            plt.Line2D(
                [0],
                [0],
                marker="D",
                color="w",
                markerfacecolor="white",
                markeredgecolor="#333333",
                markersize=8,
                label="Mean",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9.5,
    )

    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    ax.set_ylim(ymin - 0.28 * span, ymax + 0.05 * span)
    for i, g in enumerate(groups):
        s = uptime_stats[g]
        ax.text(
            i + 1,
            ymin - 0.16 * span,
            f"mean {s['mean']:.3f} \u00b1 {s['std']:.3f}\nmedian={s['median']:.3f}\np25={s['p25']:.3f}   p75={s['p75']:.3f}",
            ha="center",
            va="top",
            fontsize=9,
            color="#333333",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="white",
                edgecolor="#DDDDDD",
                linewidth=0.8,
            ),
        )

    # --------------------------------------------------------------------
    # Right panel: success rate
    # --------------------------------------------------------------------
    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    ax2.grid(axis="y", color="#DDDDDD", linewidth=0.8, zorder=0)
    ax2.set_axisbelow(True)

    bars = ax2.bar(
        groups,
        success_rates,
        color=colors,
        width=0.55,
        edgecolor="white",
        linewidth=1.2,
        zorder=3,
    )
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Success rate", fontsize=12)
    ax2.set_title("Success rate by experiment type", fontsize=12.5, pad=12)
    for spine in ("top", "right"):
        ax2.spines[spine].set_visible(False)

    for bar, sr in zip(bars, success_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            sr + 0.03,
            f"{sr:.0%}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#222222",
        )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    return uptime_stats


def print_summary(name, summary):
    print(f"\n=== {name} ===")
    for i, r in enumerate(summary["per_experiment"]):
        line = f"  Experiment {i + 1}: uptime={r['uptime']:.3f}, success={r['success']}"
        if "longest_hold_s" in r:
            line += f", longest_hold={r['longest_hold_s']:.2f}s"
        if "per_segment_success" in r:
            line += f", per_segment={r['per_segment_success']}"
        print(line)
    print(f"  --> Success rate: {summary['success_rate']:.2%}")
    print(f"  --> Mean uptime:  {summary['mean_uptime']:.3f}")


if __name__ == "__main__":
    nd_summary = analyze_file(NON_DISTURBANCE_PKL, disturbance=False)
    print_summary("Non-disturbance experiments", nd_summary)

    d_summary = analyze_file(
        DISTURBANCE_PKL,
        disturbance=True,
        disturbance_times_all=DISTURBANCE_TIMES,
    )
    print_summary("Disturbance experiments", d_summary)

    uptime_stats = plot_summary(nd_summary, d_summary)
    print("\n=== Uptime stats (mean, median, std, p25, p75) ===")
    for group, stats in uptime_stats.items():
        print(f"  {group}: {stats}")
