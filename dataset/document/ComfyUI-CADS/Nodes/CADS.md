---
tags:
- Animation
---

# CADS
## Documentation
- Class name: `CADS`
- Category: `utils`
- Output node: `False`

The CADS node is designed to apply conditional adaptive diffusion steps to a given model, enhancing its ability to generate or modify content based on specified conditions. It dynamically adjusts the diffusion process based on the progression of steps and conditions, aiming to improve the quality and relevance of the generated outputs.
## Input types
### Required
- **`model`**
    - The model to which the CADS adjustments will be applied. It serves as the base for the adaptive diffusion process, determining the initial state and capabilities of the generation or modification task.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`noise_scale`**
    - Defines the scale of noise to be added during the diffusion process. This parameter influences the degree of randomness and variation introduced at each step, affecting the diversity and novelty of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`t1`**
    - The lower threshold for the adaptive scaling factor, marking the beginning of the transition phase in the diffusion process. It helps in controlling the blend between original content and generated noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`t2`**
    - The upper threshold for the adaptive scaling factor, indicating the end of the transition phase in the diffusion process. This parameter is crucial for fine-tuning the balance between preserving original details and introducing new elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`rescale`**
    - Optional parameter that allows for the rescaling of the output after noise addition, aiming to maintain the overall distribution of the generated content. It can help in preserving the quality and consistency of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_step`**
    - Optional parameter that specifies the starting step of the adaptive diffusion process. It can be used to resume or customize the progression of the diffusion, tailoring it to specific needs or preferences.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`total_steps`**
    - Optional parameter that defines the total number of steps in the diffusion process. It sets the boundary for the adaptation and progression, influencing the depth and thoroughness of the content generation or modification.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`apply_to`**
    - Optional parameter that determines the specific part of the model to which the CADS adjustments are applied, such as conditioning or unconditioned components. It allows for targeted modifications, enhancing the relevance and effectiveness of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`key`**
    - Optional parameter that specifies the key for selecting the target component within the model for noise addition. It enables precise control over the diffusion process, focusing on specific areas or features.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`noise_type`**
    - Optional parameter that defines the type of noise to be added during the diffusion process. It offers the ability to customize the nature of the introduced randomness, affecting the texture and characteristics of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model after applying the CADS adjustments. It represents the enhanced version of the original model, equipped with adaptive diffusion capabilities for improved content generation or modification.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [SelfAttentionGuidance](../../Comfy/Nodes/SelfAttentionGuidance.md)



## Source code
```python
class CADS:
    current_step = 0
    last_sigma = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_scale": ("FLOAT", {"min": -5.0, "max": 5.0, "step": 0.01, "default": 0.25}),
                "t1": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.6}),
                "t2": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.9}),
            },
            "optional": {
                "rescale": ("FLOAT", {"min": 0.0, "max": 1.0, "step": 0.01, "default": 0.0}),
                "start_step": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "total_steps": ("INT", {"min": -1, "max": 10000, "default": -1}),
                "apply_to": (["uncond", "cond", "both"],),
                "key": (["y", "c_crossattn"],),
                "noise_type": (["normal", "uniform", "exponential"],),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "do"

    CATEGORY = "utils"

    def do(
        self,
        model,
        noise_scale,
        t1,
        t2,
        rescale=0.0,
        start_step=-1,
        total_steps=-1,
        apply_to="both",
        key="y",
        noise_type="normal",
    ):
        previous_wrapper = model.model_options.get("model_function_wrapper")

        im = model.model.model_sampling
        self.current_step = start_step
        self.last_sigma = None

        skip = None
        if apply_to == "cond":
            skip = UNCOND
        elif apply_to == "uncond":
            skip = COND

        def cads_gamma(sigma):
            if start_step >= total_steps:
                ts = im.timestep(sigma[0])
                t = round(ts.item() / 999.0, 2)
            else:
                sigma_max = sigma.max().item()
                if self.last_sigma is not None and sigma_max > self.last_sigma:
                    self.current_step = start_step
                t = 1.0 - min(1.0, max(self.current_step / total_steps, 0.0))
                self.current_step += 1
                self.last_sigma = sigma_max

            if t <= t1:
                r = 1.0
            elif t >= t2:
                r = 0.0
            else:
                r = (t2 - t) / (t2 - t1)
            return r

        def cads_noise(gamma, y):
            if y is None:
                return None
            noise = generate_noise(y, noise_type=noise_type)
            gamma = torch.tensor(gamma).to(y)
            psi = rescale
            if psi != 0:
                y_mean, y_std = y.mean(), y.std()
            y = gamma.sqrt().item() * y + noise_scale * (1 - gamma).sqrt().item() * noise
            # FIXME: does this work at all like it's supposed to?
            if psi != 0:
                y_scaled = (y - y.mean()) / y.std() * y_std + y_mean
                if not y_scaled.isnan().any():
                    y = psi * y_scaled + (1 - psi) * y
                else:
                    print("Warning, NaNs during rescale")
            return y

        def apply_cads(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            cond_or_uncond = args["cond_or_uncond"]
            c = args["c"]

            if noise_scale != 0.0:
                noise_target = c.get(key, c["c_crossattn"])
                gamma = cads_gamma(timestep)
                for i in range(noise_target.size(dim=0)):
                    if cond_or_uncond[i % len(cond_or_uncond)] == skip:
                        continue
                    noise_target[i] = cads_noise(gamma, noise_target[i])

            if previous_wrapper:
                return previous_wrapper(apply_model, args)

            return apply_model(input_x, timestep, **c)

        m = model.clone()
        m.set_model_unet_function_wrapper(apply_cads)

        return (m,)

```
