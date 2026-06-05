import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from models import Critic, GaussianActor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rollout GIF for a saved WPO run.")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to a saved WPO run directory containing config.json, actor.pt, and critic.pt.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. Defaults to <run-dir>/rollout.gif.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Rollout seed.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum number of rollout steps.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="GIF playback frames per second.",
    )
    return parser.parse_args()


def load_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def overlay_text(frame: np.ndarray, lines: list[str]) -> Image.Image:
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    x = 8
    y = 8
    line_height = 14

    # Simple translucent background substitute using solid rectangles.
    max_width = max(draw.textlength(line, font=font) for line in lines)
    box_height = 6 + line_height * len(lines)
    draw.rectangle((4, 4, 12 + max_width, 8 + box_height), fill=(0, 0, 0))

    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += line_height

    return image


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir)

    method = str(cfg.get("method", "")).upper()
    if method != "WPO":
        raise ValueError(f"Expected a WPO run, but config.json reports method={method!r}.")

    env_id = cfg.get("env_id", "Pendulum-v1")
    hidden_units = int(cfg.get("hidden_units", 256))
    min_std = float(cfg.get("min_std", 1e-3))
    max_std = float(cfg.get("max_std", float("inf")))
    device = torch.device(cfg.get("device", "cpu"))

    env = gym.make(env_id, render_mode="rgb_array")
    obs, _ = env.reset(seed=args.seed)
    env.action_space.seed(args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    action_low = env.action_space.low.astype(np.float32)
    action_high = env.action_space.high.astype(np.float32)

    actor = GaussianActor(obs_dim, act_dim, hidden_units, action_low, action_high, min_std, max_std).to(device)
    critic = Critic(obs_dim, act_dim, hidden_units).to(device)

    actor_path = run_dir / "actor.pt"
    critic_path = run_dir / "critic.pt"
    if not actor_path.exists() or not critic_path.exists():
        raise FileNotFoundError(f"Missing actor.pt or critic.pt in {run_dir}")

    actor.load_state_dict(torch.load(actor_path, map_location=device))
    critic.load_state_dict(torch.load(critic_path, map_location=device))
    actor.eval()
    critic.eval()

    frames: list[Image.Image] = []
    episode_return = 0.0

    for step in range(1, args.max_steps + 1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        mean, std = actor(obs_t)
        action_t = torch.clamp(mean, actor.action_low, actor.action_high)
        q_pred = critic(obs_t, action_t).item()
        action = action_t.detach().cpu().numpy()[0]

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_return += float(reward)

        frame = env.render()
        action_scalar = float(np.asarray(action).reshape(-1)[0])
        sigma_scalar = float(std.mean().item())
        lines = [
            f"{env_id} | WPO rollout",
            f"step: {step}",
            f"action: {action_scalar:+.3f}",
            f"sigma: {sigma_scalar:.4f}",
            f"critic Q: {q_pred:+.3f}",
            f"return: {episode_return:+.2f}",
        ]
        frames.append(overlay_text(frame, lines))

        obs = next_obs
        if terminated or truncated:
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames were generated during rollout.")

    output_path = Path(args.output) if args.output is not None else run_dir / "rollout.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(round(1000 / max(1, args.fps))))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"Saved rollout GIF to: {output_path}")


if __name__ == "__main__":
    main()
