import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WPO toy mixture-of-Gaussians experiment.")

    parser.add_argument("--method", type=str, default="WPO", choices=["PG", "NPG", "WPO"],
                        help="Optimization method.")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Monte Carlo batch size.")
    parser.add_argument("--lr", type=float, default=0.003,
                        help="Learning rate.")
    parser.add_argument("--num-iters", type=int, default=10000,
                        help="Number of optimization iterations.")

    parser.add_argument("--num-components", type=int, default=2,
                        help="Number of Gaussian mixture components.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--dtype", type=str, default="float64", choices=["float32", "float64"],
                        help="Torch dtype.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device, e.g. cpu or cuda.")

    parser.add_argument("--init-sigma", type=float, default=10.0,
                        help="Initial standard deviation for all components.")
    parser.add_argument("--init-means", type=float, nargs="*", default=None,
                        help="Initial means. If omitted and K=2, uses [-1, 1].")
    parser.add_argument("--init-alpha", type=float, default=0.0,
                        help="Initial logit value for all mixture weights.")

    parser.add_argument("--fisher-damping", type=float, default=1e-2,
                        help="Diagonal damping added to the Fisher matrix for NPG/WPO.")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="If positive, clips the update direction to this L2 norm.")

    parser.add_argument("--out-dir", type=str, default="./results/toy_examples",
                        help="Directory for logs.")

    return parser.parse_args()


def q_value(a: torch.Tensor) -> torch.Tensor:
    return -(a ** 4) / 100.0 + a ** 2


def q_grad(a: torch.Tensor) -> torch.Tensor:
    return 2.0 * a - (a ** 3) / 25.0


def normal_log_prob(a: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    # a: (M,), mu/sigma: (K,)
    # return: (M, K)
    a_col = a[:, None]
    return (
        -0.5 * ((a_col - mu[None, :]) / sigma[None, :]) ** 2
        - torch.log(sigma[None, :])
        - 0.5 * math.log(2.0 * math.pi)
    )


def log_prob_mixture(
    a: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    # return: (M,)
    sigma = torch.exp(log_sigma)
    log_rho = torch.log_softmax(alpha, dim=0)
    comp_log_probs = log_rho[None, :] + normal_log_prob(a, mu, sigma)
    return torch.logsumexp(comp_log_probs, dim=1)


@torch.no_grad()
def sample_policy(
    batch_size: int,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    sigma = torch.exp(log_sigma)
    rho = torch.softmax(alpha, dim=0)
    comp = torch.multinomial(rho, num_samples=batch_size, replacement=True)
    eps = torch.randn(batch_size, dtype=mu.dtype, device=mu.device)
    return mu[comp] + sigma[comp] * eps


def flatten_jacobian(jac_tuple: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """
    torch.autograd.functional.jacobian returns one tensor per input parameter.
    Each tensor has shape output_shape + input_shape.

    Here the output is shape (M,), and each parameter is shape (K,), so each
    Jacobian block has shape (M, K). Concatenating gives shape (M, 3K).
    """
    return torch.cat([j.reshape(j.shape[0], -1) for j in jac_tuple], dim=1)


def score_matrix(
    a: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample score vectors:

        grad_theta log pi_theta(a_m)

    for theta = (mu_1,...,mu_K, log_sigma_1,...,log_sigma_K, alpha_1,...,alpha_K).

    Return shape: (M, 3K).
    """
    def fn(mu_in: torch.Tensor, log_sigma_in: torch.Tensor, alpha_in: torch.Tensor) -> torch.Tensor:
        return log_prob_mixture(a, mu_in, log_sigma_in, alpha_in)

    jac = torch.autograd.functional.jacobian(
        fn, (mu, log_sigma, alpha), create_graph=False, strict=False
    )
    return flatten_jacobian(jac)


def wpo_cross_matrix(
    a: torch.Tensor,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """
    Compute per-sample cross derivative vectors:

        grad_theta grad_a log pi_theta(a_m)

    for theta = (mu_1,...,mu_K, log_sigma_1,...,log_sigma_K, alpha_1,...,alpha_K).

    Return shape: (M, 3K).
    """
    a_req = a.detach().clone().requires_grad_(True)

    def fn(mu_in: torch.Tensor, log_sigma_in: torch.Tensor, alpha_in: torch.Tensor) -> torch.Tensor:
        logp = log_prob_mixture(a_req, mu_in, log_sigma_in, alpha_in)
        grad_a = torch.autograd.grad(
            logp.sum(), a_req, create_graph=True, retain_graph=True
        )[0]
        return grad_a

    jac = torch.autograd.functional.jacobian(
        fn, (mu, log_sigma, alpha), create_graph=False, strict=False
    )
    return flatten_jacobian(jac)


def fisher_matrix(scores: torch.Tensor, damping: float) -> torch.Tensor:
    dim = scores.shape[1]
    fisher = scores.T @ scores / scores.shape[0]
    fisher = fisher + damping * torch.eye(dim, dtype=scores.dtype, device=scores.device)
    return fisher


def solve_fisher(fisher: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(fisher, vector)
    except RuntimeError:
        return torch.linalg.pinv(fisher) @ vector


def split_delta(delta: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_mu = delta[:k]
    delta_log_sigma = delta[k:2 * k]
    delta_alpha = delta[2 * k:3 * k]
    return delta_mu, delta_log_sigma, delta_alpha


@torch.no_grad()
def summarize_params(
    iteration: int,
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    alpha: torch.Tensor,
    objective_est: float,
    delta_norm: float,
) -> Dict[str, float]:
    sigma = torch.exp(log_sigma)
    rho = torch.softmax(alpha, dim=0)
    row: Dict[str, float] = {
        "iter": int(iteration),
        "objective_est": float(objective_est),
        "delta_norm": float(delta_norm),
    }
    for i in range(mu.numel()):
        row[f"mu_{i}"] = float(mu[i].detach().cpu())
        row[f"sigma_{i}"] = float(sigma[i].detach().cpu())
        row[f"alpha_{i}"] = float(alpha[i].detach().cpu())
        row[f"rho_{i}"] = float(rho[i].detach().cpu())
    return row


def make_initial_params(args: argparse.Namespace, dtype: torch.dtype, device: torch.device):
    k = args.num_components

    if args.init_means is None:
        if k == 2:
            init_means = [-1.0, 1.0]
        else:
            init_means = np.linspace(-1.0, 1.0, k).tolist()
    else:
        if len(args.init_means) != k:
            raise ValueError(
                f"--init-means must have length {k}, but got {len(args.init_means)}."
            )
        init_means = args.init_means

    mu = torch.tensor(init_means, dtype=dtype, device=device, requires_grad=True)
    log_sigma = torch.full(
        (k,), math.log(args.init_sigma), dtype=dtype, device=device, requires_grad=True
    )
    alpha = torch.full((k,), args.init_alpha, dtype=dtype, device=device, requires_grad=True)
    return mu, log_sigma, alpha


def main() -> None:
    args = parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dir = out_root / f"{args.method}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    mu, log_sigma, alpha = make_initial_params(args, dtype=dtype, device=device)
    k = args.num_components

    history = []

    # Log initial parameters.
    with torch.no_grad():
        a0 = sample_policy(args.batch_size, mu, log_sigma, alpha)
        obj0 = q_value(a0).mean().item()
        history.append(summarize_params(0, mu, log_sigma, alpha, obj0, 0.0))

    pbar = tqdm(range(1, args.num_iters + 1), desc=f"Toy {args.method}", dynamic_ncols=True)

    for it in pbar:
        # Fresh Monte Carlo samples from current policy.
        a = sample_policy(args.batch_size, mu.detach(), log_sigma.detach(), alpha.detach())

        # Reattach parameters as leaf tensors for autodiff in this iteration.
        mu = mu.detach().clone().requires_grad_(True)
        log_sigma = log_sigma.detach().clone().requires_grad_(True)
        alpha = alpha.detach().clone().requires_grad_(True)

        scores = score_matrix(a, mu, log_sigma, alpha)
        q = q_value(a).detach()
        pg_vector = (q[:, None] * scores).mean(dim=0)

        if args.method == "PG":
            delta = pg_vector
        else:
            fisher = fisher_matrix(scores, args.fisher_damping)

            if args.method == "NPG":
                delta = solve_fisher(fisher, pg_vector)
            elif args.method == "WPO":
                cross = wpo_cross_matrix(a, mu, log_sigma, alpha)
                qprime = q_grad(a).detach()
                wpo_vector = (qprime[:, None] * cross).mean(dim=0)
                delta = solve_fisher(fisher, wpo_vector)
            else:
                raise ValueError(f"Unknown method: {args.method}")

        if args.grad_clip > 0.0:
            delta_norm = torch.linalg.norm(delta)
            if delta_norm > args.grad_clip:
                delta = delta * (args.grad_clip / (delta_norm + 1e-12))

        delta_mu, delta_log_sigma, delta_alpha = split_delta(delta, k)

        with torch.no_grad():
            mu = mu + args.lr * delta_mu
            log_sigma = log_sigma + args.lr * delta_log_sigma
            alpha = alpha + args.lr * delta_alpha

            # Estimate objective after the update.
            a_eval = sample_policy(args.batch_size, mu, log_sigma, alpha)
            objective_est = q_value(a_eval).mean().item()
            delta_norm_value = torch.linalg.norm(delta).item()

            row = summarize_params(it, mu, log_sigma, alpha, objective_est, delta_norm_value)
            history.append(row)

            rho = torch.softmax(alpha, dim=0)
            sigma = torch.exp(log_sigma)
            pbar.set_postfix({
                "J": f"{objective_est:.3f}",
                "mu": np.array2string(mu.detach().cpu().numpy(), precision=2),
                "sigma": np.array2string(sigma.detach().cpu().numpy(), precision=2),
                "rho": np.array2string(rho.detach().cpu().numpy(), precision=2),
            })

    # Save NPZ for easier plotting
    fieldnames = list(history[0].keys())
    npz_dict = {key: np.array([row[key] for row in history]) for key in fieldnames}
    np.savez(run_dir / "params_history.npz", **npz_dict)

    print(f"\nSaved logs to: {run_dir}")


if __name__ == "__main__":
    main()
