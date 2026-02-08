from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from env.snake_env import SnakeEnv


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_agent_config(raw: Dict[str, Any]) -> DQNConfig:
    return DQNConfig(**raw)


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    agent_cfg = make_agent_config(cfg.get("agent", {}))

    out_dir = Path(train_cfg.get("output_dir", "outputs"))
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = SnakeEnv(**env_cfg)
    agent = DQNAgent(agent_cfg, device=train_cfg.get("device"))

    episodes = int(train_cfg.get("episodes", 1000))
    max_steps = int(train_cfg.get("max_steps_per_episode", 2000))
    log_interval = int(train_cfg.get("log_interval", 20))
    checkpoint_interval = int(train_cfg.get("checkpoint_interval", 100))

    metrics_path = log_dir / "train_metrics.csv"
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "score", "total_reward", "avg_loss", "epsilon", "steps"])

        scores = []
        for ep in range(1, episodes + 1):
            state = env.reset()
            total_reward = 0.0
            losses = []
            score = 0
            steps = 0

            for _ in range(max_steps):
                action = agent.select_action(state, explore=True)
                next_state, reward, done, info = env.step(action)
                agent.remember(state, action, reward, next_state, done)
                loss = agent.learn()
                if loss is not None:
                    losses.append(loss)

                state = next_state
                total_reward += reward
                score = int(info["score"])
                steps = int(info["steps"])
                if done:
                    break

            agent.decay_epsilon()
            if ep % agent.cfg.target_update_every == 0:
                agent.update_target()

            avg_loss = mean(losses) if losses else 0.0
            writer.writerow([ep, score, total_reward, avg_loss, agent.epsilon, steps])
            scores.append(score)

            if ep % log_interval == 0:
                recent = scores[-log_interval:]
                print(
                    f"Episode {ep:5d} | avg_score={mean(recent):6.2f} | "
                    f"best={max(scores):3d} | epsilon={agent.epsilon:.3f}"
                )

            if ep % checkpoint_interval == 0:
                agent.save(ckpt_dir / f"dqn_ep{ep}.pt")
                agent.save(ckpt_dir / "latest.pt")

        agent.save(ckpt_dir / "latest.pt")

    print(f"Training finished. Metrics: {metrics_path}")
    print(f"Checkpoints: {ckpt_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DQN agent for Snake.")
    parser.add_argument("--config", type=str, default="configs/dqn.yaml", help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.config)
