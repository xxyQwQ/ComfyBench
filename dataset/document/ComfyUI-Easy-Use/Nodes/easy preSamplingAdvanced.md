---
tags:
- Sampling
---

# PreSampling (Advanced)
## Documentation
- Class name: `easy preSamplingAdvanced`
- Category: `EasyUse/PreSampling`
- Output node: `True`

This node is designed to configure advanced pre-sampling settings for image generation processes. It allows for detailed customization of the sampling process, enabling users to fine-tune parameters such as noise levels, scheduler types, denoise settings, and seed values to achieve desired image qualities and variations. The node serves as a crucial component for preparing the image generation pipeline, offering flexibility and control over the pre-sampling phase.
## Input types
### Required
- **`pipe`**
    - The pipeline object that is being configured for the pre-sampling process. It is essential for defining the context and settings for the image generation workflow.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`steps`**
    - Specifies the number of steps to be executed in the pre-sampling process. This parameter directly influences the granularity and quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configures the conditioning factor for the generation process, affecting the influence of the conditioning on the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Determines the sampling algorithm to be used, allowing for selection from a variety of predefined samplers to optimize the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler algorithm to be used during the pre-sampling process. This parameter allows for the selection from a predefined list of schedulers, including custom ones, to control the sampling behavior and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at_step`**
    - Defines the starting step for the pre-sampling process, allowing for control over the initial state of image generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Sets the ending step for the pre-sampling process, determining the final state of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`add_noise`**
    - Enables or disables the addition of noise during the pre-sampling process, affecting the texture and details of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`seed`**
    - Sets the seed value for the random number generator used in the pre-sampling process. This ensures reproducibility of results by initializing the generator to a known state.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`return_with_leftover_noise`**
    - Determines whether the generated images should retain some level of noise, offering a balance between clarity and naturalness.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
### Optional
- **`image_to_latent`**
    - Optional input for providing an image to be converted into a latent representation, enabling direct manipulation of the latent space.
    - Comfy dtype: `IMAGE`
    - Python dtype: `object`
- **`latent`**
    - Optional input for providing a latent representation, offering an advanced level of customization by directly manipulating the latent space prior to sampling.
    - Comfy dtype: `LATENT`
    - Python dtype: `object`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs a configured pipeline object, encapsulating all the specified pre-sampling settings and ready for integration into the image generation workflow.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class samplerSettingsAdvanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS + new_schedulers,),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "add_noise": (["enable", "disable"],),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     "return_with_leftover_noise": (["disable", "enable"], ),
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, start_at_step, end_at_step, add_noise, seed, return_with_leftover_noise, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            _, height, width, _ = image_to_latent.shape
            if height == 1 and width == 1:
                samples = pipe["samples"]
                images = pipe["images"]
            else:
                samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
                samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
                images = image_to_latent
        elif latent is not None:
            samples = latent
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        new_pipe = {
            "model": pipe['model'],
            "positive": pipe['positive'],
            "negative": pipe['negative'],
            "vae": pipe['vae'],
            "clip": pipe['clip'],

            "samples": samples,
            "images": images,
            "seed": seed,

            "loader_settings": {
                **pipe["loader_settings"],
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "start_step": start_at_step,
                "last_step": end_at_step,
                "denoise": 1.0,
                "add_noise": add_noise,
                "force_full_denoise": force_full_denoise
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

```
