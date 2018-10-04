import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from game import Game


class ColorFlood(gym.Env):
    def __init__(self, size=6):
        self.size = size

        self.action_space = spaces.Discrete(size)
        self.observation_space = spaces.Box(low=1, high=6, shape=(size, size))

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

        self.game.change(action, start=0)

        done = self.game.is_over()

        if done:
            reward = 1
        else:
            reward = 0

        return self.observation, reward, done, {}

    def reset(self):
        self.game = Game(size=self.size)
        return self.observation

    @property
    def observation(self):
        return np.copy(self.game.main_borad)

if __name__ == "__main__":
    from random import randint

    env = ColorFlood()
    done = False
    while not done:
        obs, reward, done, info = env.step(randint(0, 5))
        print(obs, reward)