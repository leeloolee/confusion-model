# Copyright 2022 The KerasCV Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras

from .__internal__.layers.attention_block import (  # noqa: E501
    AttentionBlock,
)
from .__internal__.layers.padded_conv1d import (
    PaddedConv1D,
)
from .__internal__.layers.resnet_block import (
    ResnetBlock,
)


class ImageEncoder(keras.Sequential):
    """ImageEncoder is the VAE Encoder for StableDiffusion."""

    def __init__(self, img_height=8, img_width=128, download_weights=False):
        super().__init__(
            [
                keras.layers.Input((img_height, img_width, 1)),
                PaddedConv1D(128, 3, padding=(0, 1)),
                ResnetBlock(128),
                ResnetBlock(128),
                PaddedConv1D(128, 3, padding=(0, 1), strides=2),
                ResnetBlock(256),
                ResnetBlock(256),
                PaddedConv1D(256, 3, padding=(0, 1), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                PaddedConv1D(512, 3, padding=(0, 1), strides=2),
                ResnetBlock(512),
                ResnetBlock(512),
                ResnetBlock(512),
                AttentionBlock(512),
                ResnetBlock(512),
                keras.layers.GroupNormalization(epsilon=1e-5),
                keras.layers.Activation("swish"),
                PaddedConv1D(2, 3, padding=(0, 1)), # 8?
                PaddedConv1D(2, 1, padding= False), # 8?
                # TODO(lukewood): can this be refactored to be a Rescaling
                #  layer? Perhaps some sort of rescale and gather?
                #  Either way, we may need a lambda to gather the first 4
                #  dimensions.
                keras.layers.Lambda(lambda x: x[..., :1] * 0.18215),
            ]
        )

        # if download_weights:
        #     image_encoder_weights_fpath = keras.utils.get_file(
        #         origin="https://huggingface.co/fchollet/stable-diffusion/resolve/main/vae_encoder.h5",  # noqa: E501
        #         file_hash="c60fb220a40d090e0f86a6ab4c312d113e115c87c40ff75d11ffcf380aab7ebb",  # noqa: E501
        #     )
        #     self.load_weights(image_encoder_weights_fpath)
