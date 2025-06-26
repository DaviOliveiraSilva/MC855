import numpy as np
import gym
from gym import spaces

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()

        # Estado: [temp, alt, wind_speed, wind_dir, area_cov, pos_x, pos_y]
        self.observation_space = spaces.Box(low=0, high=500, shape=(7,), dtype=np.float32)

        # Ação: [delta_x, delta_y]
        self.action_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)

        self.reset()

    def step(self, action):
        dx, dy = action
        self.pos_x = np.clip(self.pos_x + dx, 0, 500)
        self.pos_y = np.clip(self.pos_y + dy, 0, 500)
        self.area_covered += np.random.uniform(0.5, 2.0)

        state = np.array([
            self.temperature,
            self.altitude,
            self.wind_speed,
            self.wind_dir,
            self.area_covered,
            self.pos_x,
            self.pos_y
        ], dtype=np.float32)

        reward = 1.0 - (self.wind_speed / 20) - (self.temperature / 50)

        done = self.area_covered >= 100

        return state, reward, done, {}

    def reset(self):
        self.temperature = np.random.uniform(20, 40)
        self.altitude = np.random.uniform(100, 300)
        self.wind_speed = np.random.uniform(0, 15)
        self.wind_dir = np.random.uniform(0, 360)
        self.area_covered = 0.0
        self.pos_x = np.random.uniform(0, 500)
        self.pos_y = np.random.uniform(0, 500)

        return np.array([
            self.temperature,
            self.altitude,
            self.wind_speed,
            self.wind_dir,
            self.area_covered,
            self.pos_x,
            self.pos_y
        ], dtype=np.float32)
