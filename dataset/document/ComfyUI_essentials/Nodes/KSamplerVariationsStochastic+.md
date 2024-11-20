---
tags:
- Sampling
---

# 🔧 KSampler Stochastic Variations
## Documentation
- Class name: `KSamplerVariationsStochastic+`
- Category: `essentials/sampling`
- Output node: `False`

This node specializes in generating stochastic variations of latent images using a specific sampling technique. It leverages randomness and a defined stochastic process to introduce variations, aiming to enhance the diversity of generated images while maintaining their core characteristics.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model used for sampling. It plays a crucial role in determining the quality and characteristics of the generated variations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latent_image`**
    - The latent_image parameter represents the initial latent representation of an image. It serves as the starting point for generating variations, influencing the final outcome.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`noise_seed`**
    - The noise_seed parameter controls the randomness of the variation noise. It ensures reproducibility of the variations by initializing the random noise generator.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The steps parameter defines the number of sampling steps. It affects the degree of variation and the quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The cfg parameter adjusts the conditioning factor, influencing the strength of the applied variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler`**
    - The sampler parameter selects the specific sampling algorithm used for generating variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter determines the scheduling algorithm for the sampling process, affecting the progression of variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - The positive parameter provides positive conditioning to guide the variation generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative parameter provides negative conditioning to steer the variation generation away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`variation_seed`**
    - The variation_seed parameter specifically controls the randomness of the variation process, distinct from the initial noise seed.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
- **`variation_strength`**
    - The variation_strength parameter adjusts the intensity of the applied variations, affecting the distinctiveness of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cfg_scale`**
    - The cfg_scale parameter scales the conditioning factor for the variation process, modifying the impact of conditioning on the variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - This output represents the latent image after applying stochastic variations, reflecting the enhanced diversity and creativity of the sampling process.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerVariationsStochastic:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                    "model": ("MODEL",),
                    "latent_image": ("LATENT", ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 25, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                    #"variation_sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                    "cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.05, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    def execute(self, model, latent_image, noise_seed, steps, cfg, sampler, scheduler, positive, negative, variation_seed, variation_strength, cfg_scale, variation_sampler="dpmpp_2m_sde"):
        # Stage 1: composition sampler
        force_full_denoise = False # return with leftover noise = "enable"
        disable_noise = False # add noise = "enable"

        end_at_step = max(int(steps * (1-variation_strength)), 1)
        start_at_step = 0

        work_latent = latent_image.copy()
        batch_size = work_latent["samples"].shape[0]
        work_latent["samples"] = work_latent["samples"][0].unsqueeze(0)

        stage1 = common_ksampler(model, noise_seed, steps, cfg, sampler, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]

        if batch_size > 1:
            stage1["samples"] = stage1["samples"].clone().repeat(batch_size, 1, 1, 1)

        # Stage 2: variation sampler
        force_full_denoise = True
        disable_noise = True
        cfg = max(cfg * cfg_scale, 1.0)
        start_at_step = end_at_step
        end_at_step = steps

        return common_ksampler(model, variation_seed, steps, cfg, variation_sampler, scheduler, positive, negative, stage1, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

```
