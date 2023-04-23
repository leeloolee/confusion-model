from ddpg import OUNoise, update_target, get_actor, get_critic
from envs import ComparingEnvironment, TrajectoryEnvWrapper
import numpy as np
import tensorflow as tf
import gymnasium as gym
import matplotlib.pyplot as plt
import functools
from stable_diffusion import StableDiffusion

class Buffer:
    def __init__(self, num_states, num_actions, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch #+ gamma * target_critic(
            #    [next_state_batch, target_actions], training=True
            # )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


std_dev = 0.2
# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001
total_episodes = 100
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005
problem = "Pendulum-v1"


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state), axis=0)
    noise = noise_object.sample()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise
    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, -1.0, 1.0)

    return legal_action

# Takes about 4 min to train
if __name__ == '__main__':
    _env = gym.make(problem)
    observation_space = _env.observation_space
    action_space = _env.action_space
    ou_noise = OUNoise(action_space.shape[0])
    horizon = 200
    actor_model = get_actor(observation_space.shape[0], action_space.shape[0], horizon)
    critic_model = get_critic(action_space.shape[0], observation_space.shape[0])
    target_actor = get_actor(observation_space.shape[0], action_space.shape[0], horizon)
    target_critic = get_critic(action_space.shape[0], observation_space.shape[0])
    # Making the weights equal initially
    target_actor.set_weights(actor_model.get_weights())
    target_critic.set_weights(critic_model.get_weights())

    critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
    actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

    buffer = Buffer(_env.observation_space.shape[0], _env.action_space.shape[0]*horizon, 50000, 64)

    # To store reward history of each episode
    ep_reward_list = []
    # To store average reward history of last few episodes
    avg_reward_list = []
    env = ComparingEnvironment('Pendulum-v1', wrapper=TrajectoryEnvWrapper)
    ############### Stable Diffusion ##############
    # stable_diffusion = StableDiffusion(img_height=512, img_width=512, jit_compile=False,)
    for ep in range(total_episodes):
        target_policy = functools.partial(policy, noise_object=OUNoise(action_space.shape[0], horizon=200))
        state0 = env.reset(default_policy=target_policy)
        episodic_reward = 0

        while True:
            # Uncomment this to see the Actor in action
            # But not in a python notebook.
            tf_prev_state = tf.expand_dims(tf.convert_to_tensor(state0), 0)
            # action = policy(tf_prev_state, ou_noise)

            # Recieve state and reward from environment.
            state, reward, done, info = env.step(action)

            # state0, action tuple,
            action = np.concatenate(action, axis=0).reshape(-1)
            buffer.record((state0, action, reward, state0)) # should change state0 to next state, but how
            episodic_reward += reward

            # buffer.learn()
            update_target(target_actor.variables, actor_model.variables, tau)
            update_target(target_critic.variables, critic_model.variables, tau)

            # End this episode when `done` is True
            if done:
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)

    # Plotting graph
    # Episodes versus Avg. Rewards
    plt.plot(avg_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Avg. Epsiodic Reward")
    plt.show()