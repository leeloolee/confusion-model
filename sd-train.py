from textwrap import wrap
import os

from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from stable_diffusion.diffusion_model import DiffusionModel
from stable_diffusion.image_encoder import ImageEncoder
from stable_diffusion.noise_scheduler import NoiseScheduler
from stable_diffusion.text_encoder import TextEncoder
from tensorflow import keras
from trainer import Trainer
import pickle
import openai
RESOLUTION = 256
AUTO = tf.data.AUTOTUNE
openai.api_key = "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"
openai_model = "text-embedding-ada-002"
tokenized_texts = openai.Embedding.create(model=openai_model, input=" ")["data"][0]["embedding"]
open_file = './data/hopper-medium-replay-v2.pkl'
batch_size = 4
with open(open_file, 'rb') as f:
    data = pickle.load(f)
obs_act_size = data[0]['observations'].shape[1] + data[0]['actions'].shape[1]
obs_act_data = np.array([], ).reshape(0, 128, obs_act_size,1)

for num, trajectory in enumerate(data[:300]): ## fix ?? 왜 1000dl dksldi
    for num in range(len(trajectory["actions"])-127):
        obs = trajectory["observations"][num:num+128]
        act = trajectory["actions"][num:num+128]
        obs_act = np.concatenate([obs, act],axis=1).reshape(1, 128, obs_act_size,1)
        breakpoint()
        obs_act_data = np.concatenate([obs_act_data, obs_act], axis = 0) # if ys.size else
obs_act_data = np.zeros((10000, obs_act_size,128, 1)) #pseudo data
dataset = tf.data.Dataset.from_tensor_slices((obs_act_data))
dataset = dataset.shuffle(30000)

tokenizer = SimpleTokenizer()
PADDING_TOKEN = 49407
MAX_PROMPT_LENGTH = 77
def process_text(caption):
    tokens = tokenizer.encode(caption)
    tokens = tokens + [PADDING_TOKEN] * (MAX_PROMPT_LENGTH- len(tokens))
    return np.array(tokens)

def run_text_encoder(image_batch):
    return (
        image_batch,
        process_text(""),
        openai.Embedding.create(model=openai_model, input="")["data"][0]["embedding"],
    )
def prepare_dict(trajectory_batch, token_batch, encoded_text_batch):
    return {
        "images": trajectory_batch,
        "tokens": token_batch,
        "encoded_text": encoded_text_batch,
    }
dataset = dataset.map(run_text_encoder, num_parallel_calls=AUTO)
dataset = dataset.map(prepare_dict, num_parallel_calls=AUTO).batch(batch_size).prefetch(AUTO)


# Take a sample batch and investigate.
sample_batch = next(iter(dataset))

for k in sample_batch:
    print(k, sample_batch[k].shape)

# Enable mixed-precision training if the underlying GPU has tensor cores.
USE_MP = False
if USE_MP:
    keras.mixed_precision.set_global_policy("mixed_float16")

image_encoder = ImageEncoder(14, 128)
# breakpoint()
diffusion_ft_trainer = Trainer(
    diffusion_model=DiffusionModel(14, 128, MAX_PROMPT_LENGTH),
    # Remove the top layer from the encoder, which cuts off the variance and only
    # returns the mean.
    vae=tf.keras.Model(
        image_encoder.input,
        image_encoder.layers[-2].output,
    ),
    noise_scheduler=NoiseScheduler(),
    use_mixed_precision=USE_MP,
)


lr = 1e-5
beta_1, beta_2 = 0.9, 0.999
weight_decay = (1e-2,)
epsilon = 1e-08

optimizer = tf.keras.optimizers.experimental.AdamW(
    learning_rate=lr,
    weight_decay=weight_decay,
    beta_1=beta_1,
    beta_2=beta_2,
    epsilon=epsilon,
)
diffusion_ft_trainer.compile(optimizer=optimizer, loss="mse")

epochs = 1
ckpt_path = "finetuned_stable_diffusion.h5"
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
    ckpt_path,
    save_weights_only=True,
    monitor="loss",
    mode="min",
)
diffusion_ft_trainer.fit(dataset, epochs=epochs, callbacks=[ckpt_callback])