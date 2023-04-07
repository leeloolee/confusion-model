from matplotlib import pyplot as plt, animation
import gymnasium as gym
from gymnasium.spaces import Tuple, Box
import numpy as np
import ray


class ComparingTrajectoryEnvironment(gym.Env):
    """


    """
    # metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
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
        return (obs1, obs2)

    def step(self, action):
        action1, action2 = action
        obs1, reward1, _, done1, info1 = self.env1.step(action1)
        obs2, reward2, _, done2, info2 = self.env2.step(action2)

        combined_obs = (obs1, obs2)
        combined_reward = reward1>reward2
        combined_done = done1 or done2
        combined_info = {"info1": info1, "info2": info2}

        return combined_obs, combined_reward, combined_done, combined_info

    def close(self):
        self.env1.close()
        self.env2.close()

    @ray.remote
    def render_env1(self):
        self.env1.render()

    @ray.remote
    def render_env2(self):
        self.env2.render()

    def render(self, mode='rgb_array'):
        if mode == 'human':
            ray.get([self.render_env1.remote(), self.render_env2.remote()])
        elif mode == 'rgb_array':
            return (self.env1.render(), self.env2.render())

def create_anim(frames, dpi, fps):
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=dpi)
    patch = plt.imshow(frames[0])
    def setup():
        plt.axis('off')
    def animate(i):
        patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=setup, frames=len(frames), interval=fps)
    return anim

def save_anim(frames, filename, dpi=72, fps=50):
    anim = create_anim(frames, dpi, fps)
    anim.save(filename)



if __name__ == '__main__':
    env1 = gym.make('CartPole-v0', render_mode='rgb_array')
    env2 = gym.make('CartPole-v0', render_mode='rgb_array')
    env = ComparingTrajectoryEnvironment(env1, env2)
    total_episodes = 1
    frames1 = []
    frames2 = []

    for episode in range(total_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done:
            frame1, frame2 = env.render(mode='rgb_array')
            frames1.append(frame1)
            frames2.append(frame2)
            action = env.action_space.sample() # 무작위 행동 선택
            state, reward, done, info = env.step(action)
            step += 1

            if done:
                print(f"에피소드: {episode + 1}, 스텝 수: {step}")
                break

    env.close()
    filename1 = 'cartpole-1.mp4'
    filename2 = 'cartpole-2.mp4'
    save_anim(frames1, filename=filename1)
    save_anim(frames2, filename=filename2)