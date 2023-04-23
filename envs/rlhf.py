import gymnasium as gym
import numpy as np
from unittest.mock import Mock
class ActionNormalizer(gym.ActionWrapper):
    """Rescale and relocate the actions."""

    def action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (-1, 1) to (low, high)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = action * scale_factor + reloc_factor
        action = np.clip(action, low, high)

        return action

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """Change the range (low, high) to (-1, 1)."""
        low = self.action_space.low
        high = self.action_space.high

        scale_factor = (high - low) / 2
        reloc_factor = high - scale_factor

        action = (action - reloc_factor) / scale_factor
        action = np.clip(action, -1.0, 1.0)

        return action

class ComparingEnvironment(gym.Env):
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self, env_name, default_policy= None,  wrapper=None, **kwargs):
        super().__init__()
        self.env_name = env_name
        self.env1 = ActionNormalizer(gym.make(env_name, render_mode = "rgb_array")) #Action Normalizer?
        self.env2 = ActionNormalizer(gym.make(env_name, render_mode = "rgb_array"))
        if default_policy:
            self.default_policy = default_policy
        else:
            mock = Mock()
            mock.__call__ = Mock(return_value=self.env1.action_space.sample())
            self.default_policy = mock
        if wrapper:
            self.env1 = wrapper(self.env1)
            self.env2 = wrapper(self.env2)
        # Combine action spaces
        self.action_space = self.env1.action_space # Tuple((env1.action_space, env2.action_space))
        self.observation_space = self.env1.observation_space

        # Combine observation spaces
        # if isinstance(env1.observation_space, Box) and isinstance(env2.observation_space, Box):
        #     low = np.concatenate((env1.observation_space.low, env2.observation_space.low))
        #     high = np.concatenate((env1.observation_space.high, env2.observation_space.high))
        #     self.observation_space = Box(low=low, high=high, dtype=np.float32)
        # else:
        #     self.observation_space = Tuple((env1.observation_space, env2.observation_space))

    def reset(self,default_policy= None, seed=0):
        if default_policy is not None:
            self.default_policy = default_policy
        self.done1 = False
        self.done2 = False
        self.reward1_list = []
        self.reward2_list = []
        self.obs1 = self.env1.reset()[0]
        self.obs2 = self.env2.reset()[0]
        return self.obs1

    def step(self, action):
        if not self.done1:
            self.obs1, reward1, _, self.done1, info1 = self.env1.step(action)
            self.obs1 = self.obs1[0]
            self.reward1_list.append(reward1)
        else:
            self.obs1 = np.zero_like(self.obs1)

        if not self.done2:
            action2 =  self.default_policy(self.obs2.reshape(1,-1))
            self.obs2, reward2, _, self.done2, info2 = self.env2.step(action2)
            self.obs2 = self.obs2[0]
            self.reward2_list.append(reward2)
        else:
            self.obs2 = np.zero_like(self.obs2)

        combined_done = self.done1 and self.done2
        combined_reward = 0
        self.return1 = sum(self.reward1_list)
        self.return2 = sum(self.reward2_list)
        if combined_done:
            if self.return1>self.return2: combined_reward = 1
            elif  self.return1==self.return2: combined_reward = 0.5
            else: combined_reward = 0
        combined_info = {"info1": info1, "info2": info2}

        return self.obs1, combined_reward, combined_done, combined_info

    def close(self):
        self.env1.close()
        self.env2.close()

    def render(self):
        return (self.env1.render(), self.env2.render())
