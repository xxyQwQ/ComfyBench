---
tags:
- Sampling
---

# KSampler (Advanced/pipe)
## Documentation
- Class name: `ImpactKSamplerAdvancedBasicPipe`
- Category: `ImpactPack/sampling`
- Output node: `False`

This node provides an advanced sampling mechanism tailored for generating or modifying images through a sophisticated pipeline. It leverages a combination of models and custom sampling strategies to enhance the quality and control of the output images, focusing on achieving specific visual impacts or adjustments.
## Input types
### Required
- **`basic_pipe`**
    - Represents the core components required for sampling, including models and configurations, essential for the advanced sampling process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
- **`add_noise`**
    - Indicates whether additional noise should be introduced in the sampling process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Provides a seed for the random noise generation, ensuring reproducibility in the sampling outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Determines the total number of steps in the sampling process, affecting the detail and quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Specifies the classifier free guidance value, influencing the direction and strength of the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampler to be used, allowing for customization of the sampling strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the noise schedule to be applied during the sampling process, affecting the progression of noise addition or removal.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`latent_image`**
    - The initial latent image to be modified or enhanced through the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`start_at_step`**
    - Specifies the starting step for the sampling process within the defined range of steps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Defines the ending step for the sampling, allowing for partial application of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`return_with_leftover_noise`**
    - Decides whether the output latent image should retain some of the introduced noise, affecting the final image texture.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`scheduler_func_opt`**
    - [OPTIONAL] Specifies an alternative noise schedule generation function, offering flexibility beyond the predefined scheduler.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `function`
## Output types
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The core components used in the sampling process, returned unchanged.
    - Python dtype: `tuple`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The resulting latent image after the sampling process, ready for further processing or conversion to an image.
    - Python dtype: `torch.Tensor`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model used in the sampling process, facilitating transformations between latent and image spaces.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvancedBasicPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"basic_pipe": ("BASIC_PIPE",),
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enable", "label_off": "disable"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                     "scheduler": (core.SCHEDULERS, ),
                     "latent_image": ("LATENT", ),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enable", "label_off": "disable"}),
                     },
                "optional":
                    {
                        "scheduler_func_opt": ("SCHEDULER_FUNC", ),
                    }
                }

    TOOLTIPS = {
        "input": {
            "basic_pipe": "basic_pipe input for sampling",
            "add_noise": "Whether to add noise",
            "noise_seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "latent_image": "input latent image",
            "start_at_step": "The starting step of the sampling to be applied at this node within the range of 'steps'.",
            "end_at_step": "The step at which sampling applied at this node will stop within the range of steps (if greater than steps, sampling will continue only up to steps).",
            "return_with_leftover_noise": "Whether to return the latent with noise remaining if the noise has not been completely removed according to the noise schedule, or to completely remove the noise before returning it.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("passthrough input basic_pipe", "result latent", "VAE in basic_pipe")
    }

    RETURN_TYPES = ("BASIC_PIPE", "LATENT", "VAE")
    FUNCTION = "sample"

    CATEGORY = "ImpactPack/sampling"

    @staticmethod
    def sample(basic_pipe, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, scheduler_func_opt=None):
        model, clip, vae, positive, negative = basic_pipe

        latent = separated_sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, scheduler_func=scheduler_func_opt)
        return basic_pipe, latent, vae

```
