from stable_diffusion import StableDiffusion

## 이건 obs+act, horizon, 1
policy_model = StableDiffusion(
    obs_act =4, horizon = 128
)
images_original = policy_model.text_to_image("HI", batch_size=1)
breakpoint()