import numpy as np
from gymnasium import Env, spaces

class DroneEnv(Env):
    def __init__(self, map_size=100):
        super().__init__()
        self.map_size = map_size
        self.coverage_map = np.zeros((map_size, map_size))

        self.observation_space = spaces.Dict({
            "pos_x": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            "pos_y": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            "coverage": spaces.Box(0, 1.0, shape=(), dtype=np.float32),
            "battery": spaces.Box(0, np.inf, shape=(), dtype=np.float32),
            "wind_speed": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
            "wind_angle": spaces.Box(0, 360, shape=(), dtype=np.int32),
            "altitude": spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
        })

        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

        self.reset()
    

    def _get_obs(self):
        return {
            "pos_x": self.pos[0],
            "pos_y": self.pos[1],
            "coverage": self.coverage_map[self.pos[0] % self.map_size, self.pos[1] % self.map_size],
            "battery": self.battery,
            "wind_speed": self.wind_speed,
            "wind_angle": self.wind_angle,
            "altitude": self.altitude,
        }

    def _get_info(self):
        return {
            "total_covered": np.count_nonzero(self.coverage_map)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.coverage_map[:] = 0
        self.pos = np.random.randint(0, self.map_size, size=(2,))
        self.battery = 1.0
        self.wind_speed = 0.0
        self.wind_angle = 0
        self.altitude = 0.0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_reward(self):
        # normalizacao da velocidade do vento
        wind_speed_normal = self.wind_speed / 100
        # se já mapeou regiao
        already_covered = self.coverage_map[self.pos[0], self.pos[1]]

        # Recompensa adaptada
        alpha = 1.0      # incentivo por nova área
        beta = 10.0       # penalização por cair / drone no chão
        gamma = 0.3      # penalização por vento
        delta = 0.5      # penalização por bateria baixa

        reward = 0.0

        reward += alpha * (1 - already_covered)
        reward -= gamma * (wind_speed_normal * np.cos(np.radians([self.wind_angle]))) # vento * cos(direcao do vento)

        if self.battery < 0.15:
            reward -= delta * (1 - self.battery)
        
        if self.altitude <= 0:
            reward -= beta

        return reward

    def step(self, action):
        dx = int(action[0])
        dy = int(action[1])
        new_x = np.clip(self.pos[0] + dx, 0, self.map_size - 1)
        new_y = np.clip(self.pos[1] + dy, 0, self.map_size - 1)
        self.pos = np.array([new_x, new_y])

        # self.altitude = self.altitude
        self.battery = self.battery - 0.01  # Simulate battery consumption
        # self.wind_speed =
        # self.wind_angle = params["wind_angle"]

        reward = self._get_reward()
        self.coverage_map[new_x, new_y] = 1.0
        done = bool(self.battery <= 0 or np.all(self.coverage_map))

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, done, False, info
        # {
        #     "observation": obs,
        #     "reward": reward,
        #     "terminated": done,
        #     "truncated": False,
        #     "info": info
        # }