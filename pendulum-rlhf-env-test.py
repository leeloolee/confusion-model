import gymnasium as gym
import numpy as np
from envs import ComparingEnvironment, trjenv

def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name)
    policy = env.action_space.sample
    env = ComparingEnvironment(env_name, default_policy=None, wrapper= trjenv.TrajectoryEnvWrapper)
    done, done1, done2 = False, False, False
    state0 = env.reset()
    env.render()
    while not done: # for _ in range(1000):
        action1 = policy()
        trajectory, reward, done, info = env.step(action1)

if __name__ == '__main__':
    main()