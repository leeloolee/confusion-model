import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from stable_diffusion import StableDiffusion
def get_actor(obs_dim, action_dim, horizon):
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = keras.Input(shape=(obs_dim,))
    ### pseudo model
    out = layers.Dense(10, activation="relu")(inputs)
    outputs = layers.Dense(action_dim*horizon, activation="tanh", kernel_initializer=last_init)(out)
    outputs = tf.keras.layers.Reshape((horizon, action_dim))(outputs)
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic(obs_dim, act_dim):
    # State as input
    inputs = layers.Input(shape=(None, obs_dim+act_dim))
    out = tf.keras.layers.LSTM(6)(inputs)
    out = layers.Dense(32, activation="relu")(out)

    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model(inputs, outputs)

    return model

if __name__ == '__main__':
    input1 = tf.zeros((1,10,4)) # pendulum action space - 1, state space - 3, horizon - 10
    input2 = tf.zeros((1,3)) # pendulum action space - 1, state space - 3, horizon - 10
    actor = get_actor(3,)
    critic = get_critic(3, 1)
    print("actor", actor(input2),"critic", critic(input1))
    keras.utils.plot_model(actor, "my_first_model_with_shape_info.png", show_shapes=True)
