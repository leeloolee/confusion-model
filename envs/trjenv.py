import gymnasium as gym
from gymnasium.spaces import Sequence
class TrajectoryEnvWrapper(gym.Wrapper):
    """
    recieve trajectory and this envrionment runs trajectorie.
    ---
    env = TrajectoryEnvWrapper(HalfCheetahEnv())
    env.step([[1,0,0,0,0,1], [1,0,0,0,0,1], ..., [0,0,0,0,0,0]])
    """
    def __init__(self, env, horizon = 200):
        super().__init__(env)
        self.env = env
        self.action_space = Sequence(self.env.action_space)

    def step(self, actions):
        rewards = 0
        obs_list = []
        action_list = []
        done = False
        for action in actions:
            if not done:
                obs, reward, done, _, info = self.env.step(action)
                action_list.append(action)
                obs_list.append(obs)
                rewards += reward   # sum of reward
                done = True
        return obs_list, rewards, done, _, action_list