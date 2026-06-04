import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("PG", "NPG", "WPO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize toy MoG optimization runs.")
    parser.add_argument(
        "--results-root",
        type=str,
        default="./results/toy_examples",
        help="Root directory containing run folders with config.json and params_history.npz.",
    )
    parser.add_argument(
        "--num-obj-points",
        type=int,
        default=2048,
        help="Number of shared Monte Carlo points per iteration for objective estimation.",
    )
    parser.add_argument(
        "--num-policy-samples",
        type=int,
        default=200,
        help="Number of blue policy samples to draw per iteration in trajectory plots.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Plot one trajectory point every N iterations to control rendering cost.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used for visualization sampling.",
    )
    parser.add_argument(
        "--a-min",
        type=float,
        default=-10.0,
        help="Minimum action value shown on x/y axes.",
    )
    parser.add_argument(
        "--a-max",
        type=float,
        default=10.0,
        help="Maximum action value shown on x/y axes.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./results/toy_examples/wpo_toy_vis.png",
        help="Path to save the generated figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, show the figure window.",
    )
    return parser.parse_args()


def q_value(a: np.ndarray) -> np.ndarray:
    return -(a ** 4) / 100.0 + a ** 2


def normal_pdf(a: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    z = (a[:, None] - mu[None, :]) / sigma[None, :]
    coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma[None, :])
    return coeff * np.exp(-0.5 * z ** 2)


def mixture_pdf(a: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    rho = np.exp(alpha - np.max(alpha))
    rho = rho / np.sum(rho)
    return np.sum(rho[None, :] * normal_pdf(a, mu, sigma), axis=1)


def discover_runs(results_root: Path) -> Dict[str, Path]:
    run_by_method: Dict[str, Tuple[Path, float]] = {}

    for cfg_path in results_root.rglob("config.json"):
        run_dir = cfg_path.parent
        npz_path = run_dir / "params_history.npz"
        if not npz_path.exists():
            continue

        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        method = str(cfg.get("method", "")).upper()
        if method not in METHODS:
            continue

        mtime = npz_path.stat().st_mtime
        prev = run_by_method.get(method)
        if prev is None or mtime > prev[1]:
            run_by_method[method] = (run_dir, mtime)

    return {m: run_by_method[m][0] for m in run_by_method}


def _stack_prefixed(npz: np.lib.npyio.NpzFile, prefix: str, k: int) -> np.ndarray:
    parts = [np.asarray(npz[f"{prefix}_{i}"]) for i in range(k)]
    return np.stack(parts, axis=1)


def load_run(run_dir: Path) -> Dict[str, np.ndarray]:
    npz_path = run_dir / "params_history.npz"
    with np.load(npz_path) as npz:
        keys = set(npz.files)
        mu_keys = sorted([k for k in keys if k.startswith("mu_")], key=lambda x: int(x.split("_")[1]))
        k = len(mu_keys)
        if k == 0:
            raise ValueError(f"No mu_i entries found in {npz_path}.")

        it = np.asarray(npz["iter"]).astype(int)
        obj_logged = np.asarray(npz["objective_est"]).astype(float)
        mu = _stack_prefixed(npz, "mu", k).astype(float)
        sigma = _stack_prefixed(npz, "sigma", k).astype(float)
        alpha = _stack_prefixed(npz, "alpha", k).astype(float)
        rho = _stack_prefixed(npz, "rho", k).astype(float)

    return {
        "iter": it,
        "objective_logged": obj_logged,
        "mu": mu,
        "sigma": sigma,
        "alpha": alpha,
        "rho": rho,
    }


def sample_from_policy_params(
    mu: np.ndarray,
    sigma: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    cdf = np.cumsum(rho)
    cdf[-1] = 1.0
    comp = np.searchsorted(cdf, u, side="right")
    comp = np.minimum(comp, len(mu) - 1)
    return mu[comp] + sigma[comp] * z


def estimate_objective_curves(
    runs: Dict[str, Dict[str, np.ndarray]],
    num_points: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    t = len(next(iter(runs.values()))["iter"])
    rng = np.random.default_rng(seed)

    # Shared random numbers across methods so objective curves are directly comparable.
    shared_u = rng.random((t, num_points))
    shared_z = rng.standard_normal((t, num_points))

    out: Dict[str, np.ndarray] = {}
    for method in METHODS:
        run = runs[method]
        curve = np.zeros(t, dtype=float)
        for i in range(t):
            a = sample_from_policy_params(
                run["mu"][i],
                run["sigma"][i],
                run["rho"][i],
                shared_u[i],
                shared_z[i],
            )
            curve[i] = float(np.mean(q_value(a)))
        out[method] = curve
    return out


def make_output_paths(output_path: Path) -> Dict[str, Path]:
    base_dir = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix if output_path.suffix else ".png"
    return {
        "value_policy": base_dir / f"{stem}_value_policy{suffix}",
        "log_objective": base_dir / f"{stem}_log_objective{suffix}",
        "PG": base_dir / f"{stem}_trajectory_pg{suffix}",
        "NPG": base_dir / f"{stem}_trajectory_npg{suffix}",
        "WPO": base_dir / f"{stem}_trajectory_wpo{suffix}",
    }


def make_value_policy_figure(
    runs: Dict[str, Dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> plt.Figure:
    ref = runs["PG"]

    a_grid = np.linspace(args.a_min, args.a_max, 1200)
    q_grid = q_value(a_grid)
    pi0 = mixture_pdf(a_grid, ref["mu"][0], ref["sigma"][0], ref["alpha"][0])

    fig, ax0 = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax0.plot(a_grid, q_grid, color="black", lw=2.3)
    ax0.set_xlabel("a", fontsize=16, style="italic")
    ax0.set_ylabel("Q(a)", fontsize=22, rotation=0, labelpad=18, style="italic")

    ax0_t = ax0.twinx()
    ax0_t.fill_between(a_grid, pi0, color="#2F5FD0", alpha=0.24)
    ax0_t.plot(a_grid, pi0, color="#2F5FD0", alpha=0.6, lw=1.2)
    ax0_t.set_ylabel("pi(a)", fontsize=22, color="#2F5FD0", rotation=0, labelpad=20, style="italic")
    ax0_t.tick_params(axis="y", colors="#2F5FD0")
    ax0.set_title("MoG Value Function", fontsize=18)
    ax0.set_xlim(args.a_min, args.a_max)
    ax0.grid(alpha=0.2, linestyle=":")
    return fig


def make_objective_figure(
    runs: Dict[str, Dict[str, np.ndarray]],
    obj_estimates: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    colors = {"PG": "#E4572E", "NPG": "#3A86FF", "WPO": "#2A9D8F"}
    for method in METHODS:
        run = runs[method]
        log_obj = np.log(np.clip(obj_estimates[method], 1e-12, None))
        ax1.plot(run["iter"], log_obj, lw=2.0, color=colors[method], label=method)
    ax1.set_title(f"Log Objective ({args.num_obj_points} shared points/iter)", fontsize=18)
    ax1.set_xlabel("Iterations", fontsize=16)
    ax1.set_ylabel("Log Objective", fontsize=16)
    ax1.grid(alpha=0.25, linestyle=":")
    ax1.legend(frameon=True, fontsize=12)
    return fig


def make_trajectory_figure(
    run: Dict[str, np.ndarray],
    method: str,
    args: argparse.Namespace,
    seed: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    mode = math.sqrt(50.0)

    rng = np.random.default_rng(seed)
    idx = np.arange(0, len(run["iter"]), max(1, args.plot_every))
    it = run["iter"][idx]
    mu = run["mu"][idx]
    sigma = run["sigma"][idx]
    rho = run["rho"][idx]

    u = rng.random((len(it), args.num_policy_samples))
    z = rng.standard_normal((len(it), args.num_policy_samples))
    x = np.repeat(it[:, None], args.num_policy_samples, axis=1)
    y = np.zeros_like(x, dtype=float)
    for i in range(len(it)):
        y[i] = sample_from_policy_params(mu[i], sigma[i], rho[i], u[i], z[i])

    ax.plot([it[0], it[-1]], [mode, mode], "k--", lw=1.8, label=r"$a^*$")
    ax.plot([it[0], it[-1]], [-mode, -mode], "k--", lw=1.8)
    ax.scatter(x.ravel(), y.ravel(), s=3, color="#3A86FF", alpha=0.7, linewidths=0, label=r"$a\sim\pi$")
    for j in range(mu.shape[1]):
        label = r"$\mu_i$" if j == 0 else None
        ax.plot(it, mu[:, j], color="#E63946", lw=1.7, label=label)

    ax.set_title(method, fontsize=18)
    ax.set_xlabel("Iterations", fontsize=14)
    ax.set_ylabel("a", fontsize=14, rotation=0, labelpad=10, style="italic")
    ax.set_xlim(it[0], it[-1])
    ax.set_ylim(args.a_min, args.a_max)
    ax.grid(alpha=0.2, linestyle=":")
    if method == "WPO":
        ax.legend(frameon=True, fontsize=12)
    return fig


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    if not results_root.exists():
        raise FileNotFoundError(f"Results root does not exist: {results_root}")

    discovered = discover_runs(results_root)
    missing = [m for m in METHODS if m not in discovered]
    if missing:
        found_text = ", ".join(f"{m}:{str(discovered[m])}" for m in sorted(discovered)) or "none"
        raise RuntimeError(
            "Could not find runs for all methods. Missing: "
            + ", ".join(missing)
            + f". Found: {found_text}.\n"
            + "Expected each run folder to contain both config.json and params_history.npz."
        )

    runs = {m: load_run(discovered[m]) for m in METHODS}

    # Align methods to a common iteration horizon for fair plotting.
    n_steps = min(len(runs[m]["iter"]) for m in METHODS)
    for m in METHODS:
        for key in ("iter", "objective_logged", "mu", "sigma", "alpha", "rho"):
            runs[m][key] = runs[m][key][:n_steps]

    obj_estimates = estimate_objective_curves(
        runs=runs,
        num_points=args.num_obj_points,
        seed=args.seed,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_paths = make_output_paths(out_path)

    fig_value = make_value_policy_figure(runs, args)
    fig_value.savefig(output_paths["value_policy"], dpi=args.dpi)

    fig_objective = make_objective_figure(runs, obj_estimates, args)
    fig_objective.savefig(output_paths["log_objective"], dpi=args.dpi)

    figs_traj = []
    for idx, method in enumerate(METHODS):
        fig_traj = make_trajectory_figure(runs[method], method, args, seed=args.seed + 1000 + idx)
        fig_traj.savefig(output_paths[method], dpi=args.dpi)
        figs_traj.append(fig_traj)

    for name, path in output_paths.items():
        print(f"Saved {name} figure to: {path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig_value)
        plt.close(fig_objective)
        for fig_traj in figs_traj:
            plt.close(fig_traj)


if __name__ == "__main__":
    main()
