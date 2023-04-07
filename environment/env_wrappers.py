"""
recieve two trajectory and this envrionment runs two trajectories in parallel.
evaluate the performance of the two trajectories.
then return the reward of result of evaluation if current policy is more improved then 1 and 0 otherwise.
"""

import random
from gym.envs.mujoco.half_cheetah_v4 import HalfCheetahEnv
import gymnasium as gym
from gymnasium.spaces import Sequence, Tuple, Box
import numpy as np
import ray

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
        self.action_space = Sequence(self.env.action_space)

    def step(self, actions):
        rewards = 0
        for action in actions:
            if not terminated:
                obs, reward, terminated, info = self.env.step(action)
                rewards += reward   # sum of reward
                terminated = True
        return obs, rewards, terminated, info

class ComparingTrajectoryEnvironment(gym.Env):
    """


    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, env1, env2):
        super().__init__()
        self.env1 = env1
        self.env2 = env2

        # Combine action spaces
        self.action_space = Tuple((env1.action_space, env2.action_space))

        # Combine observation spaces
        if isinstance(env1.observation_space, Box) and isinstance(env2.observation_space, Box):
            low = np.concatenate((env1.observation_space.low, env2.observation_space.low))
            high = np.concatenate((env1.observation_space.high, env2.observation_space.high))
            self.observation_space = Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = Tuple((env1.observation_space, env2.observation_space))
        if self.render_mode == "human":
            ray.init()
            self.env1

    def reset(self):
        obs1 = self.env1.reset()
        obs2 = self.env2.reset()
        obs1.reshape(1, -1)
        obs2.reshape(1, -1)
        breakpoint()
        return np.concatenate((obs1, obs2))

    def step(self, action):
        action1, action2 = action
        obs1, reward1, done1, info1 = self.env1.step(action1)
        obs2, reward2, done2, info2 = self.env2.step(action2)

        combined_obs = np.concatenate((obs1, obs2))
        combined_reward = reward1 + reward2
        combined_done = done1 or done2
        combined_info = {"info1": info1, "info2": info2}

        return combined_obs, combined_reward, combined_done, combined_info

    def close(self):
        self.env1.close()
        self.env2.close()

    @ray.remote
    def render_env1(self, mode='human'):
        self.env1.render(mode)

    @ray.remote
    def render_env2(self, mode='human'):
        self.env2.render(mode)

    def render(self, mode='rgb_array'):
        if mode == 'human':
            ray.get([self.render_env1.remote(mode), self.render_env2.remote(mode)])
        elif mode == 'rgb_array':
            return self.env1.render(mode), self.env2.render(mode)



if __name__ == '__main__':
    env1 = gym.make('CartPole-v0')
    env2 = gym.make('CartPole-v0')
    env = ComparingTrajectoryEnvironment(env1, env2)
    total_episodes = 5

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done:
            env.render()
            action = env.action_space.sample() # 무작위 행동 선택
            state, reward, done, info = env.step(action)
            step += 1

            if done:
                print(f"에피소드: {episode + 1}, 스텝 수: {step}")
                break

    env.close()