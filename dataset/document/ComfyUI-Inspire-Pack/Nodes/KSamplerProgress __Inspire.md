---
tags:
- Sampling
---

# KSampler Progress (Inspire)
## Documentation
- Class name: `KSamplerProgress __Inspire`
- Category: `InspirePack/analysis`
- Output node: `False`

The KSamplerProgress //Inspire node is designed to iteratively generate samples from a model, providing intermediate results at specified intervals. This node is particularly useful for monitoring the progress of sample generation, allowing for adjustments based on intermediate outputs.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model from which samples are to be generated. It plays a crucial role in determining the characteristics and quality of the generated samples.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`seed`**
    - The seed parameter sets the initial seed for the sampling process, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The steps parameter determines the number of steps to perform in the sample generation process, impacting the detail and quality of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The cfg parameter configures the sampling process, influencing how the model generates samples based on the provided configuration.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The sampler_name parameter identifies the specific sampling strategy to be used, affecting the generation process and the characteristics of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter allows for the specification of a scheduling strategy to control the sampling process over time, potentially improving the quality of the generated samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `object`
- **`positive`**
    - The positive parameter specifies positive guidance prompts, directing the model to generate samples that align with the provided positive descriptions.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative parameter specifies negative guidance prompts, instructing the model to avoid generating samples that align with the provided negative descriptions.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent_image`**
    - The latent_image parameter provides the initial state for the sample generation process, which can be modified through the sampling steps.
    - Comfy dtype: `LATENT`
    - Python dtype: `object`
- **`denoise`**
    - The denoise parameter controls the level of denoising applied to the samples, affecting the clarity and quality of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - The noise_mode parameter defines how noise is applied during the sample generation process, affecting the diversity and quality of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`interval`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`omit_start_latent`**
    - The omit_start_latent parameter determines whether the initial latent state is included in the output, affecting the range of samples returned.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`scheduler_func_opt`**
    - The scheduler_func_opt parameter provides an optional scheduling function to further customize the sampling process, potentially enhancing the generation results.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `object`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - unknown
    - Python dtype: `unknown`
- **`progress_latent`**
    - Comfy dtype: `LATENT`
    - The progress_latent output includes intermediate samples generated during the sampling process, showcasing the progression of the sample generation.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSampler_progress(a1111_compat.KSampler_inspire):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (common.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "noise_mode": (["GPU(=A1111)", "CPU"],),
                    "interval": ("INT", {"default": 1, "min": 1, "max": 10000}),
                    "omit_start_latent": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
                    },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    CATEGORY = "InspirePack/analysis"

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("latent", "progress_latent")

    @staticmethod
    def doit(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, noise_mode, interval, omit_start_latent, scheduler_func_opt=None):
        adv_steps = int(steps / denoise)

        if omit_start_latent:
            result = []
        else:
            result = [latent_image['samples']]

        result = []

        def progress_callback(step, x0, x, total_steps):
            if (total_steps-1) != step and step % interval != 0:
                return

            x = model.model.process_latent_out(x)
            x = x.to(model_management.intermediate_device())
            result.append(x)

        latent_image, noise = a1111_compat.KSamplerAdvanced_inspire.sample(model, True, seed, adv_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, (adv_steps-steps),
                                                                           adv_steps, noise_mode, False, callback=progress_callback, scheduler_func_opt=scheduler_func_opt)

        if len(result) > 0:
            result = torch.cat(result)
            result = {'samples': result}
        else:
            result = latent_image

        return latent_image, result

```
