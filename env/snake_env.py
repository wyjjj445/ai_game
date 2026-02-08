from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import pygame
except Exception:  # pragma: no cover
    pygame = None


Coord = Tuple[int, int]


@dataclass
class StepInfo:
    score: int
    steps: int
    reason: str


class SnakeEnv:
    """Simple Snake environment with relative actions.

    Actions:
        0 -> go straight
        1 -> turn right
        2 -> turn left
    """

    RIGHT = (1, 0)
    LEFT = (-1, 0)
    UP = (0, -1)
    DOWN = (0, 1)
    CLOCKWISE = [RIGHT, DOWN, LEFT, UP]

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None,
        step_penalty: float = -0.01,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        shaping_scale: float = 0.1,
        reward_shaping: bool = True,
        max_steps_without_food_factor: int = 100,
        block_size: int = 20,
    ) -> None:
        self.width = width
        self.height = height
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.shaping_scale = shaping_scale
        self.reward_shaping = reward_shaping
        self.max_steps_without_food_factor = max_steps_without_food_factor
        self.block_size = block_size

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.snake: List[Coord] = []
        self.direction: Coord = self.RIGHT
        self.food: Coord = (0, 0)
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.done = False

        self._screen = None
        self._clock = None

    def reset(self) -> np.ndarray:
        cx, cy = self.width // 2, self.height // 2
        self.direction = self.RIGHT
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.score = 0
        self.steps = 0
        self.steps_without_food = 0
        self.done = False
        self._place_food()
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int | str]]:
        if self.done:
            raise RuntimeError("Cannot call step() after done=True. Call reset() first.")

        old_distance = self._distance_to_food(self.snake[0], self.food)
        self.steps += 1
        self.steps_without_food += 1
        self._move(action)

        reward = self.step_penalty
        reason = "running"

        head = self.snake[0]
        if self._is_collision(head):
            self.done = True
            reward = self.death_penalty
            reason = "collision"
        else:
            if head == self.food:
                self.score += 1
                self.steps_without_food = 0
                reward = self.food_reward
                reason = "food"
                self._place_food()
            else:
                self.snake.pop()
                if self.reward_shaping:
                    new_distance = self._distance_to_food(self.snake[0], self.food)
                    reward += self.shaping_scale if new_distance < old_distance else -self.shaping_scale

            max_steps_without_food = self.max_steps_without_food_factor * len(self.snake)
            if self.steps_without_food > max_steps_without_food:
                self.done = True
                reward = self.death_penalty
                reason = "starvation"

        info = StepInfo(score=self.score, steps=self.steps, reason=reason).__dict__
        return self.get_state(), float(reward), self.done, info

    def get_state(self) -> np.ndarray:
        head = self.snake[0]
        idx = self.CLOCKWISE.index(self.direction)

        dir_straight = self.CLOCKWISE[idx]
        dir_right = self.CLOCKWISE[(idx + 1) % 4]
        dir_left = self.CLOCKWISE[(idx - 1) % 4]

        point_straight = (head[0] + dir_straight[0], head[1] + dir_straight[1])
        point_right = (head[0] + dir_right[0], head[1] + dir_right[1])
        point_left = (head[0] + dir_left[0], head[1] + dir_left[1])

        state = np.array(
            [
                int(self._is_collision(point_straight)),
                int(self._is_collision(point_right)),
                int(self._is_collision(point_left)),
                int(self.direction == self.LEFT),
                int(self.direction == self.RIGHT),
                int(self.direction == self.UP),
                int(self.direction == self.DOWN),
                int(self.food[0] < head[0]),
                int(self.food[0] > head[0]),
                int(self.food[1] < head[1]),
                int(self.food[1] > head[1]),
            ],
            dtype=np.float32,
        )
        return state

    def render(self, fps: int = 12) -> None:
        if pygame is None:
            return

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((self.width * self.block_size, self.height * self.block_size))
            pygame.display.set_caption("AI Snake Demo")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self._screen.fill((20, 20, 20))
        for x, y in self.snake:
            pygame.draw.rect(
                self._screen,
                (20, 180, 80),
                (x * self.block_size, y * self.block_size, self.block_size - 1, self.block_size - 1),
            )
        fx, fy = self.food
        pygame.draw.rect(
            self._screen,
            (220, 60, 60),
            (fx * self.block_size, fy * self.block_size, self.block_size - 1, self.block_size - 1),
        )
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(fps)

    def close(self) -> None:
        if pygame is not None and self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None

    def _move(self, action: int) -> None:
        idx = self.CLOCKWISE.index(self.direction)
        if action == 0:
            new_dir = self.CLOCKWISE[idx]
        elif action == 1:
            new_dir = self.CLOCKWISE[(idx + 1) % 4]
        elif action == 2:
            new_dir = self.CLOCKWISE[(idx - 1) % 4]
        else:
            raise ValueError(f"Invalid action {action}. Expected one of [0, 1, 2].")

        self.direction = new_dir
        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        self.snake.insert(0, new_head)

    def _is_collision(self, point: Coord) -> bool:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        if point in self.snake[1:]:
            return True
        return False

    def _place_food(self) -> None:
        free_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.snake]
        if not free_cells:
            self.food = self.snake[0]
            return
        self.food = free_cells[self.rng.randint(0, len(free_cells) - 1)]

    @staticmethod
    def _distance_to_food(head: Coord, food: Coord) -> int:
        return abs(head[0] - food[0]) + abs(head[1] - food[1])
