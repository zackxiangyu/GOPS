from typing import TypeVar, Tuple

import gym
import numpy as np
from gym.utils import seeding

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class NoiseAction(gym.Wrapper):
    """
        noise_type has two type: "normal" and "uniform"
        if noise_type == "normal", noise_data means Mean and Standard deviation of Normal distribution
        if noise_type == "uniform", noise_data means Upper and Lower bounds of Uniform distribution
    """
    def __init__(self, env, noise_type, noise_data):
        super(NoiseAction, self).__init__(env)
        assert noise_type in ["normal", "uniform"]
        assert len(noise_data) == 2 and len(noise_data[0]) == env.action_space.shape[0]
        self.noise_type = noise_type
        self.noise_data = np.array(noise_data, dtype=np.float32)

    def noise_action(self, action):
        if self.noise_type is None:
            return action
        elif self.noise_type == "normal":
            return action + self.np_random.normal(loc=self.noise_data[0], scale=self.noise_data[1])
        elif self.noise_type == "uniform":
            return action + self.np_random.uniform(low=self.noise_data[0], high=self.noise_data[1])


    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        action_noised = self.noise_action(action)
        obs, r, d, info = self.env.step(action_noised)
        return obs, r, d, info

    def seed(self, seed=None):
        # generate a new seed to avoid correlation between action noise and environment randomness
        np_random, _ = seeding.np_random(seed)
        noise_seed = int(np_random.randint(2**31))
        self.np_random, noise_seed = seeding.np_random(noise_seed)
        seeds = self.env.seed(seed)
        return seeds + [noise_seed]