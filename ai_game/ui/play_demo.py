from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
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


def demo(config_path: str | Path, model_path: str | Path, fps: int) -> None:
    cfg = load_config(config_path)
    env_cfg = cfg.get("env", {})
    train_cfg = cfg.get("train", {})
    agent_cfg = DQNConfig(**cfg.get("agent", {}))

    env = SnakeEnv(**env_cfg)
    agent = DQNAgent(agent_cfg, device=train_cfg.get("device"))
    agent.load(model_path)
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state, explore=False)
        state, _, done, info = env.step(action)
        env.render(fps=fps)
        # Keep a tiny fallback delay if pygame is unavailable.
        time.sleep(0.001)

    env.close()
    print(f"Demo finished. Score={info['score']}, steps={info['steps']}, reason={info['reason']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained DQN Snake agent.")
    parser.add_argument("--config", type=str, default="configs/dqn.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--fps", type=int, default=12, help="Render FPS")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo(args.config, args.model, args.fps)
