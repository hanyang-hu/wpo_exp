import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize inverted-pendulum experiment logs.")
    parser.add_argument(
        "--result-dir", type=str, required=True,
        help="Run directory or parent directory containing multiple runs.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to write output figures. Defaults to <result-dir>.",
    )
    parser.add_argument(
        "--dpi", type=int, default=200, help="Figure DPI.",
    )
    # Seed-averaging options
    parser.add_argument(
        "--aggregate", action="store_true", default=True,
        help="Group runs by method and average over seeds (default: True).",
    )
    parser.add_argument(
        "--no-aggregate", dest="aggregate", action="store_false",
        help="Plot every run as a separate line without aggregation.",
    )
    parser.add_argument(
        "--variance", type=str, default="std",
        choices=["none", "std", "sem", "minmax"],
        help=(
            "Variance band style when --aggregate is on. "
            "'none' = mean only; 'std' = +/-1 std; "
            "'sem' = +/-1 standard error; 'minmax' = min/max envelope."
        ),
    )
    parser.add_argument(
        "--interp-steps", type=int, default=500,
        help="Number of evenly-spaced steps used for interpolation before averaging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Discovery & loading
# ---------------------------------------------------------------------------

def find_metric_files(root: Path) -> List[Path]:
    """Find all metrics.csv files at depth 1 or 2 under root."""
    if (root / "metrics.csv").exists():
        return [root / "metrics.csv"]
    found = sorted(root.glob("*/metrics.csv")) + sorted(root.glob("*/*/metrics.csv"))
    return sorted(set(found))


_SEED_RE = re.compile(r"_seed\d+", re.IGNORECASE)
_METHOD_RE = re.compile(r"^(PG|NPG|DPG|WPO)", re.IGNORECASE)
_SCALE_RE = re.compile(r"scale(\w+)", re.IGNORECASE)

# Methods that have their own intrinsic natural-gradient scaling —
# for these, the scale suffix does not change the displayed name.
_SELF_SCALED = {"WPO", "DPG", "NPG"}


def method_label(run_dir: Path) -> str:
    """Derive a clean, report-style legend label from a run directory name.

    Rules:
      - DPG / WPO / NPG            -> "DPG" / "WPO" / "NPG"  (scale irrelevant)
      - PG  + scale none           -> "PG"
            - PG  + scale wpo            -> "PG"  (WPO-only scaling does not apply to PG)
            - PG  + scale pg / all       -> "PG (scaled)"
    Falls back to the raw name (seed stripped) if the pattern is not matched.
    """
    name = run_dir.name
    method_m = _METHOD_RE.search(name)
    if method_m is None:
        return _SEED_RE.sub("", name)

    method = method_m.group(1).upper()
    if method in _SELF_SCALED:
        return method

    # PG (and any unknown method): check scale value
    scale_m = _SCALE_RE.search(name)
    scale = scale_m.group(1).lower() if scale_m else "none"
    if scale in {"none", "", "wpo"}:
        return method
    return f"{method} (scaled)"


def load_runs(metric_files: List[Path]) -> Dict[str, List[pd.DataFrame]]:
    """Return {method_label: [df, df, ...]} grouping runs that share a method."""
    groups: Dict[str, List[pd.DataFrame]] = {}
    for p in metric_files:
        df = pd.read_csv(p)
        label = method_label(p.parent)
        groups.setdefault(label, []).append(df)
    return groups


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def interpolate_series(
    steps: np.ndarray, values: np.ndarray, grid: np.ndarray
) -> np.ndarray:
    """Linear interpolation of (steps, values) onto a common grid."""
    return np.interp(grid, steps, values)


def aggregate(
    dfs: List[pd.DataFrame],
    col: str,
    interp_steps: int,
    dropna: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (grid, mean, lower, upper) arrays after interpolating all runs
    onto a common step grid.  Returns None if the column is absent in all dfs.
    """
    series_list = []
    for df in dfs:
        if col not in df.columns:
            continue
        sub = df[["step", col]].copy()
        if dropna:
            sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        series_list.append((sub["step"].to_numpy(), sub[col].to_numpy()))

    if not series_list:
        return None

    step_min = max(s[0].min() for s in series_list)
    step_max = min(s[0].max() for s in series_list)
    if step_min >= step_max:
        # Fallback: use the full union range
        step_min = min(s[0].min() for s in series_list)
        step_max = max(s[0].max() for s in series_list)

    grid = np.linspace(step_min, step_max, interp_steps)
    mat = np.stack([interpolate_series(s, v, grid) for s, v in series_list], axis=0)

    mean = mat.mean(axis=0)
    lower = mat.min(axis=0)
    upper = mat.max(axis=0)
    return grid, mean, lower, upper


def variance_bounds(
    dfs: List[pd.DataFrame],
    col: str,
    interp_steps: int,
    style: str,
    dropna: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    result = aggregate(dfs, col, interp_steps, dropna=dropna)
    if result is None:
        return None
    grid, mean, lower, upper = result

    if style == "none" or len(dfs) == 1:
        return grid, mean, mean, mean

    series_list = []
    for df in dfs:
        if col not in df.columns:
            continue
        sub = df[["step", col]].copy()
        if dropna:
            sub = sub.dropna(subset=[col])
        if sub.empty:
            continue
        series_list.append(interpolate_series(sub["step"].to_numpy(), sub[col].to_numpy(), grid))

    mat = np.stack(series_list, axis=0)
    mean = mat.mean(axis=0)

    if style == "std":
        sd = mat.std(axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(mean)
        lower, upper = mean - sd, mean + sd
    elif style == "sem":
        n = mat.shape[0]
        sem = mat.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
        lower, upper = mean - sem, mean + sem
    else:  # minmax
        lower, upper = mat.min(axis=0), mat.max(axis=0)

    return grid, mean, lower, upper


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "PG": "#E4572E",
    "NPG": "#3A86FF",
    "WPO": "#2A9D8F",
    "DPG": "#8B5CF6",
}


def color_for(label: str) -> Optional[str]:
    for key, col in METHOD_COLORS.items():
        if key.upper() in label.upper():
            return col
    return None


def plot_column(
    ax: plt.Axes,
    groups: Dict[str, List[pd.DataFrame]],
    col: str,
    ylabel: str,
    xlabel: str,
    args,
    dropna: bool = False,
) -> None:
    for label, dfs in sorted(groups.items()):
        c = color_for(label)
        if args.aggregate:
            result = variance_bounds(dfs, col, args.interp_steps, args.variance, dropna=dropna)
            if result is None:
                continue
            grid, mean, lower, upper = result
            ax.plot(grid, mean, lw=2.0, label=label, color=c)
            if args.variance != "none" and len(dfs) > 1:
                ax.fill_between(grid, lower, upper, alpha=0.18, color=c)
        else:
            for i, df in enumerate(dfs):
                if col not in df.columns:
                    continue
                sub = df[["step", col]].copy()
                if dropna:
                    sub = sub.dropna(subset=[col])
                run_label = label if i == 0 else None
                ax.plot(sub["step"], sub[col], lw=1.4, label=run_label, color=c, alpha=0.75)

    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.grid(True, alpha=0.25, linestyle=":")
    ax.legend(fontsize=9, frameon=True)


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    root = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else root

    metric_files = find_metric_files(root)
    if not metric_files:
        raise FileNotFoundError(f"No metrics.csv found under {root}")

    groups = load_runs(metric_files)

    # Figure 1: Training return
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plot_column(ax, groups, "episode_return_avg10",
                ylabel="Training Return", xlabel="Environment Steps", args=args)
    save_fig(fig, out_dir / "fig_train_return.png", args.dpi)

    # Figure 2: Eval return
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plot_column(ax, groups, "eval_return",
                ylabel="Evaluation Return", xlabel="Environment Steps",
                args=args, dropna=True)
    save_fig(fig, out_dir / "fig_eval_return.png", args.dpi)

    # Figure 3: Critic loss
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plot_column(ax, groups, "critic_loss",
                ylabel="Critic Loss", xlabel="Environment Steps", args=args)
    save_fig(fig, out_dir / "fig_critic_loss.png", args.dpi)


if __name__ == "__main__":
    main()
