import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DroneEnv(gym.Env):
    def __init__(self, map_size=50):
        super().__init__()
        self.map_size = map_size
        self.coverage_map = np.zeros((map_size, map_size))
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    def reset(self):
        self.coverage_map[:] = 0
        self.pos = np.random.randint(0, self.map_size, size=(2,))
        self.battery = 1.0
        self.wind_speed = np.random.uniform(0, 1)  # Normalizado [0, 1]
        return self._get_state()

    def _get_state(self):
        return np.array([
            self.pos[0] / self.map_size,
            self.pos[1] / self.map_size,
            self.coverage_map[self.pos[0], self.pos[1]],
            self.battery,
            np.random.uniform(0, 1),  # temperatura (normalizado)
            self.wind_speed,
            np.random.uniform(0, 1),  # altitude
            np.random.uniform(0, 1),  # direção do vento
        ], dtype=np.float32)

    def step(self, action):
        dx = int(action[0] * 2)
        dy = int(action[1] * 2)
        new_x = np.clip(self.pos[0] + dx, 0, self.map_size - 1)
        new_y = np.clip(self.pos[1] + dy, 0, self.map_size - 1)
        self.pos = np.array([new_x, new_y])

        already_covered = self.coverage_map[new_x, new_y]
        self.battery = max(0, self.battery - 0.01)

        # Recompensa adaptada
        alpha = 1.0      # incentivo por nova área
        beta = 0.3       # penalização por redundância
        gamma = 0.5      # penalização por vento
        delta = 0.4      # penalização por bateria baixa

        reward = 0.0

        if already_covered < 0.5:
            reward += alpha * (1 - already_covered)
        else:
            reward -= beta

        reward -= gamma * self.wind_speed

        if self.battery < 0.2:
            reward -= delta * (1 - self.battery)

        self.coverage_map[new_x, new_y] = 1.0
        done = self.battery <= 0

        return self._get_state(), reward, done, {}