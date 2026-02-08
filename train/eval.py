from __future__ import annotations

import argparse
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


def evaluate(config_path: str | Path, model_path: str | Path, episodes: int) -> None:
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    agent_cfg = DQNConfig(**cfg.get("agent", {}))

    env = SnakeEnv(**env_cfg)
    agent = DQNAgent(agent_cfg, device=train_cfg.get("device"))
    agent.load(model_path)
    agent.epsilon = 0.0

    scores = []
    steps_list = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        score = 0
        while not done:
            action = agent.select_action(state, explore=False)
            state, _, done, info = env.step(action)
            steps = int(info["steps"])
            score = int(info["score"])
        scores.append(score)
        steps_list.append(steps)

    print(f"Episodes: {episodes}")
    print(f"Average score: {mean(scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Average steps: {mean(steps_list):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model.")
    parser.add_argument("--config", type=str, default="configs/dqn.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Evaluation episodes")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.config, args.model, args.episodes)
