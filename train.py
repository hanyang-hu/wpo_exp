import argparse
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models import Critic, DeterministicActor, GaussianActor
from utils import (
    OUNoise,
    ReplayBuffer,
    append_csv,
    ensure_dir,
    evaluate_policy,
    hard_update,
    save_json,
    set_seed,
    soft_update,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDPG-style PG/DPG/WPO on Gymnasium Pendulum.")
    parser.add_argument("--env-id", type=str, default="Pendulum-v1",
                        help="Gymnasium environment ID.")
    parser.add_argument("--method", type=str, default="WPO", choices=["PG", "DPG", "WPO"],
                        help="Actor update method.")

    parser.add_argument("--num-iters", type=int, default=1_000,
                        help="Number of environment steps. The script also works for small values such as 1000.")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Replay-buffer minibatch size for critic and actor updates.")
    parser.add_argument("--sample-size", type=int, default=32,
                        help="Number of action samples per state for stochastic targets and stochastic actor updates.")
    parser.add_argument("--hidden-units", type=int, default=256,
                        help="Number of hidden units in the single hidden layer of actor/critic networks.")

    parser.add_argument("--buffer-size", type=int, default=100_000,
                        help="Maximum replay-buffer size.")
    parser.add_argument("--start-steps", type=int, default=None,
                        help="Number of initial environment steps using random actions. If omitted, chosen automatically from num-iters.")
    parser.add_argument("--update-after", type=int, default=None,
                        help="Start gradient updates after this many environment steps. If omitted, chosen automatically from num-iters.")
    parser.add_argument("--update-every", type=int, default=1,
                        help="Perform one gradient update every this many environment steps after update-after.")

    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor.")
    parser.add_argument("--tau", type=float, default=0.005,
                        help="Polyak averaging coefficient for target-network updates.")
    parser.add_argument("--actor-lr", type=float, default=1e-3,
                        help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=1e-3,
                        help="Critic learning rate.")
    parser.add_argument("--actor-grad-clip", type=float, default=10.0,
                        help="Actor gradient clipping norm. Set <=0 to disable.")
    parser.add_argument("--critic-grad-clip", type=float, default=10.0,
                        help="Critic gradient clipping norm. Set <=0 to disable.")

    parser.add_argument("--min-std", type=float, default=1e-3,
                        help="Minimum standard deviation for stochastic Gaussian policies.")
    parser.add_argument("--max-std", type=float, default=float("inf"),
                        help="Maximum standard deviation for stochastic Gaussian policies.")
    parser.add_argument("--gaussian-fisher-scaling", type=str, default="wpo",
                        choices=["none", "pg", "wpo", "all"],
                        help=("Apply simplified Gaussian Fisher scaling to stochastic policy-output gradients. "
                              "'wpo' reproduces the WPO setting; 'pg' applies it only to PG; "
                              "'all' applies it to both PG and WPO; 'none' disables it."))
    parser.add_argument("--pg-use-baseline", action="store_true", default=True,
                        help="For PG, subtract a Q baseline to reduce gradient variance (default: True).")
    parser.add_argument("--no-pg-use-baseline", dest="pg_use_baseline", action="store_false",
                        help="Disable the PG baseline term.")

    parser.add_argument("--ou-sigma", type=float, default=0.2,
                        help="Ornstein-Uhlenbeck exploration noise scale for DPG.")
    parser.add_argument("--ou-theta", type=float, default=0.15,
                        help="Ornstein-Uhlenbeck mean-reversion parameter for DPG.")

    parser.add_argument("--eval-every", type=int, default=None,
                        help="Evaluate every this many environment steps. If omitted, chosen automatically from num-iters.")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of episodes used for each evaluation.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device.")
    parser.add_argument("--out-dir", type=str, default="./results/inverted_pendulum",
                        help="Directory where results are stored.")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Optional run-name. If omitted, a method-based name is used.")
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Choose small-run-friendly defaults after argparse."""
    if args.num_iters <= 0:
        raise ValueError("--num-iters must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")

    # For small runs such as --num-iters 1000, the previous defaults
    # start-steps=1000 and update-after=1000 left essentially no training time.
    warmup = max(10, min(1000, args.num_iters // 10))
    if args.start_steps is None:
        args.start_steps = warmup
    else:
        args.start_steps = min(args.start_steps, max(0, args.num_iters - 1))

    if args.update_after is None:
        args.update_after = warmup
    else:
        args.update_after = min(args.update_after, max(0, args.num_iters - 1))

    if args.eval_every is None:
        args.eval_every = max(1, min(2000, args.num_iters // 5))
    else:
        args.eval_every = max(1, min(args.eval_every, args.num_iters))

    args.update_every = max(1, args.update_every)
    args.buffer_size = max(args.buffer_size, args.batch_size)
    return args


def critic_target_stochastic(
    target_actor: GaussianActor,
    target_critic: Critic,
    next_obs: torch.Tensor,
    sample_size: int,
) -> torch.Tensor:
    batch_size = next_obs.shape[0]
    with torch.no_grad():
        next_actions, _, _, _ = target_actor.sample(next_obs, sample_size=sample_size, detach_action=True)
        next_obs_rep = next_obs.unsqueeze(0).expand(sample_size, batch_size, next_obs.shape[-1])
        q_next = target_critic(
            next_obs_rep.reshape(sample_size * batch_size, -1),
            next_actions.reshape(sample_size * batch_size, -1),
        )
        return q_next.reshape(sample_size, batch_size, 1).mean(dim=0)


def dpg_actor_loss(actor: DeterministicActor, critic: Critic, obs: torch.Tensor) -> torch.Tensor:
    return -critic(obs, actor(obs)).mean()


def should_use_gaussian_fisher_scaling(args: argparse.Namespace, update_type: str) -> bool:
    update_type = update_type.lower()
    return args.gaussian_fisher_scaling in {"all", update_type}


def stochastic_actor_backward(
    actor: GaussianActor,
    critic: Critic,
    obs: torch.Tensor,
    sample_size: int,
    update_type: str,
    use_pg_baseline: bool,
    use_gaussian_fisher_scaling: bool,
    grad_clip: float,
) -> tuple[float, float]:
    """
    Backpropagate a stochastic actor update through a Gaussian policy.

    update_type='PG':
        maximizes E[Q(s,a) log pi(a|s)] with sampled actions detached.
        If use_pg_baseline=True, uses (Q - b) log pi where b is the
        per-state sample mean of Q, which reduces variance.

    update_type='WPO':
        maximizes E[grad_a log pi(a|s)^T grad_a Q(s,a)].

    If use_gaussian_fisher_scaling=True, the output-level gradients are scaled as
        grad_mu  <- sigma^2 grad_mu,
        grad_std <- 1/2 sigma^2 grad_std,
    before being backpropagated through the actor network.

    Returns:
        objective_value, actor_grad_norm
    """
    if update_type not in {"PG", "WPO"}:
        raise ValueError(f"stochastic_actor_backward only supports PG/WPO, got {update_type}.")

    actor.zero_grad(set_to_none=True)
    batch_size = obs.shape[0]
    mean, std = actor(obs)
    dist = torch.distributions.Normal(mean, std)

    actions = dist.sample((sample_size,)).detach()
    actions = torch.max(torch.min(actions, actor.action_high), actor.action_low)
    obs_rep = obs.unsqueeze(0).expand(sample_size, batch_size, obs.shape[-1])
    obs_flat = obs_rep.reshape(sample_size * batch_size, -1)
    act_flat = actions.reshape(sample_size * batch_size, -1)

    if update_type == "PG":
        logp = dist.log_prob(actions).sum(dim=-1)
        q_values = critic(obs_flat, act_flat).detach().reshape(sample_size, batch_size)
        if use_pg_baseline:
            baseline = q_values.mean(dim=0, keepdim=True)
            q_values = q_values - baseline
        objective = (q_values * logp).mean()
    else:
        act_req = act_flat.detach().requires_grad_(True)
        q_sum = critic(obs_flat, act_req).sum()
        q_grad = torch.autograd.grad(q_sum, act_req, create_graph=False, retain_graph=False)[0]
        q_grad = q_grad.detach().reshape(sample_size, batch_size, -1)

        mean_rep = mean.unsqueeze(0).expand_as(actions)
        std_rep = std.unsqueeze(0).expand_as(actions)
        grad_a_logp = (mean_rep - actions) / (std_rep.pow(2) + 1e-8)
        objective = (grad_a_logp * q_grad).sum(dim=-1).mean()

    grad_mean, grad_std = torch.autograd.grad(
        objective,
        (mean, std),
        retain_graph=True,
        create_graph=False,
        allow_unused=False,
    )

    if use_gaussian_fisher_scaling:
        grad_mean = std.pow(2).detach() * grad_mean
        grad_std = 0.5 * std.pow(2).detach() * grad_std

    # Adam minimizes gradients. Negative output gradients implement ascent.
    torch.autograd.backward(
        tensors=(mean, std),
        grad_tensors=(-grad_mean, -grad_std),
    )

    if grad_clip > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), grad_clip).item())
    else:
        total = 0.0
        for param in actor.parameters():
            if param.grad is not None:
                total += float(param.grad.detach().pow(2).sum().item())
        grad_norm = math.sqrt(total)

    return float(objective.detach().item()), grad_norm


def main() -> None:
    args = finalize_args(parse_args())
    set_seed(args.seed)
    device = torch.device(args.device)

    env = gym.make(args.env_id)
    eval_env = gym.make(args.env_id)
    obs, _ = env.reset(seed=args.seed)
    eval_env.reset(seed=args.seed + 10_000)
    env.action_space.seed(args.seed)
    eval_env.action_space.seed(args.seed + 10_000)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    run_name = args.run_name or (
        f"{args.method}_seed{args.seed}_"
        f"scale{args.gaussian_fisher_scaling}"
        f"_pgbaseline{'on' if args.pg_use_baseline else 'off'}"
    )
    run_dir = ensure_dir(Path(args.out_dir) / run_name)
    save_json(vars(args), run_dir / "config.json")

    if args.method == "DPG":
        actor = DeterministicActor(obs_dim, act_dim, args.hidden_units, action_low, action_high).to(device)
        target_actor = DeterministicActor(obs_dim, act_dim, args.hidden_units, action_low, action_high).to(device)
    else:
        actor = GaussianActor(obs_dim, act_dim, args.hidden_units, action_low, action_high, args.min_std, args.max_std).to(device)
        target_actor = GaussianActor(obs_dim, act_dim, args.hidden_units, action_low, action_high, args.min_std, args.max_std).to(device)

    critic = Critic(obs_dim, act_dim, args.hidden_units).to(device)
    target_critic = Critic(obs_dim, act_dim, args.hidden_units).to(device)
    hard_update(target_actor, actor)
    hard_update(target_critic, critic)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(obs_dim, act_dim, args.buffer_size, device=device)
    ou_noise = OUNoise(act_dim, theta=args.ou_theta, sigma=args.ou_sigma)

    metrics_path = run_dir / "metrics.csv"
    episode_return = 0.0
    episode_idx = 0
    recent_returns: list[float] = []
    last_critic_loss = math.nan
    last_actor_objective = math.nan
    last_actor_grad_norm = math.nan
    last_eval_return = math.nan

    def maybe_nan(value: float):
        return value if not math.isnan(value) else None

    pbar = tqdm(range(1, args.num_iters + 1), desc=f"{args.env_id} {args.method}", dynamic_ncols=True)
    for step in pbar:
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            if args.method == "DPG":
                with torch.no_grad():
                    action = actor(obs_t).cpu().numpy()[0] + ou_noise.sample()
            else:
                with torch.no_grad():
                    action_t, _, _, _ = actor.sample(obs_t, sample_size=1, detach_action=True)
                    action = action_t.cpu().numpy()[0]
            action = np.clip(action, action_low, action_high)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay.add(obs, action, reward, next_obs, terminated)
        episode_return += float(reward)
        obs = next_obs

        if done:
            recent_returns.append(episode_return)
            recent_returns = recent_returns[-10:]

            # Ensure the first logged episode has a valid evaluation value.
            if math.isnan(last_eval_return):
                last_eval_return = evaluate_policy(eval_env, actor, args.method, device, args.eval_episodes)

            episode_idx += 1
            row = {
                "step": step,
                "episode": episode_idx,
                "episode_return_last": recent_returns[-1],
                "episode_return_avg10": float(np.mean(recent_returns)),
                "eval_return": maybe_nan(last_eval_return),
                "critic_loss": maybe_nan(last_critic_loss),
                "actor_objective": maybe_nan(last_actor_objective),
                "actor_grad_norm": maybe_nan(last_actor_grad_norm),
                "replay_size": len(replay),
                "start_steps": args.start_steps,
                "update_after": args.update_after,
                "eval_every": args.eval_every,
            }
            if args.method != "DPG":
                with torch.no_grad():
                    obs_probe = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    _, std_probe = actor(obs_probe)
                    row.update({
                        "std_mean": float(std_probe.mean().item()),
                        "std_min": float(std_probe.min().item()),
                        "std_max": float(std_probe.max().item()),
                        "gaussian_fisher_scaling_used": should_use_gaussian_fisher_scaling(args, args.method),
                        "pg_baseline_used": bool(args.pg_use_baseline) if args.method == "PG" else False,
                    })
            append_csv(row, metrics_path)

            obs, _ = env.reset()
            ou_noise.reset()
            episode_return = 0.0

        if step >= args.update_after and len(replay) >= args.batch_size and step % args.update_every == 0:
            obs_b, act_b, rew_b, next_obs_b, done_b = replay.sample(args.batch_size)

            with torch.no_grad():
                if args.method == "DPG":
                    q_next = target_critic(next_obs_b, target_actor(next_obs_b))
                else:
                    q_next = critic_target_stochastic(target_actor, target_critic, next_obs_b, args.sample_size)
                target_q = rew_b + args.gamma * (1.0 - done_b) * q_next

            q = critic(obs_b, act_b)
            critic_loss = F.mse_loss(q, target_q)
            critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            if args.critic_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.critic_grad_clip)
            critic_opt.step()
            last_critic_loss = float(critic_loss.item())

            if args.method == "DPG":
                actor_loss = dpg_actor_loss(actor, critic, obs_b)
                actor_opt.zero_grad(set_to_none=True)
                actor_loss.backward()
                if args.actor_grad_clip > 0:
                    last_actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(actor.parameters(), args.actor_grad_clip).item())
                else:
                    last_actor_grad_norm = math.nan
                actor_opt.step()
                last_actor_objective = float((-actor_loss).detach().item())
            else:
                use_scaling = should_use_gaussian_fisher_scaling(args, args.method)
                if args.method == "PG" and args.gaussian_fisher_scaling == "wpo" and use_scaling:
                    raise RuntimeError("PG should not use Gaussian Fisher scaling when --gaussian-fisher-scaling=wpo.")
                last_actor_objective, last_actor_grad_norm = stochastic_actor_backward(
                    actor=actor,
                    critic=critic,
                    obs=obs_b,
                    sample_size=args.sample_size,
                    update_type=args.method,
                    use_pg_baseline=args.pg_use_baseline,
                    use_gaussian_fisher_scaling=use_scaling,
                    grad_clip=args.actor_grad_clip,
                )
                actor_opt.step()

            soft_update(target_actor, actor, args.tau)
            soft_update(target_critic, critic, args.tau)

        if step % args.eval_every == 0 or step == args.num_iters:
            last_eval_return = evaluate_policy(eval_env, actor, args.method, device, args.eval_episodes)

        pbar.set_postfix({
            "ep": episode_idx,
            "ret10": f"{np.mean(recent_returns):.1f}" if recent_returns else "nan",
            "eval": f"{last_eval_return:.1f}" if not math.isnan(last_eval_return) else "nan",
            "Qloss": f"{last_critic_loss:.3g}" if not math.isnan(last_critic_loss) else "nan",
        })

    torch.save(actor.state_dict(), run_dir / "actor.pt")
    torch.save(critic.state_dict(), run_dir / "critic.pt")
    print(f"\nSaved results to: {run_dir}")


if __name__ == "__main__":
    main()
