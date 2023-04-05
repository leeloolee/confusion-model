"""
recieve two trajectory and this envrionment runs two trajectories in parallel.
evaluate the performance of the two trajectories.
then return the reward of result of evaluation if current policy is more improved then 1 and 0 otherwise.
"""

import random
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import gymnasium as gym
from gymnasium.spaces import Sequence
class TrajectoryEnvWrapper(gym.Wrapper):
    """
    recieve trajectory and this envrionment runs trajectorie.

    ---
    env = TrajectoryEnvWrapper(HalfCheetahEnv())
    env.step([[1,0,0,0,0,1], [1,0,0,0,0,1], ..., [0,0,0,0,0,0]])
    """
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.action_space = Sequence(self.action_space)
    def step(self, actions):
        rewards = 0
        for action in actions:
            if not terminated:
                obs, reward, terminated, info = self.env.step(action)
                rewards += reward   # sum of reward
        return obs, rewards, terminated, info




if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    env = TrajectoryEnvWrapper(env)