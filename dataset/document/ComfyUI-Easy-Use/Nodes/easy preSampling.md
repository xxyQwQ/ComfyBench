---
tags:
- Sampling
---

# PreSampling
## Documentation
- Class name: `easy preSampling`
- Category: `EasyUse/PreSampling`
- Output node: `True`

The 'easy preSampling' node is designed to configure and initiate a pre-sampling process, optimizing the input data before it undergoes further processing or model inference. This node abstracts the complexity of pre-sampling settings, allowing users to easily adjust parameters for improved model performance or data quality.
## Input types
### Required
- **`pipe`**
    - Specifies the pipeline configuration for the pre-sampling process, determining the sequence of operations and models to be applied.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `str`
- **`steps`**
    - Defines the number of steps to be executed in the pre-sampling process, affecting the depth of sampling and refinement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Adjusts the conditioning factor, influencing the guidance and direction of the pre-sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampler algorithm to be used in the pre-sampling process, tailoring the sampling strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduler for controlling the sampling process, managing the progression through steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Sets the denoising level applied during the pre-sampling, enhancing the clarity and quality of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - Provides a seed value for the random number generator, ensuring reproducibility of the pre-sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`image_to_latent`**
    - Converts an input image to a latent representation, preparing it for further processing in the pre-sampling stage.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`latent`**
    - Specifies the initial latent space representation to be used or modified during the pre-sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The output pipeline configuration, reflecting any adjustments or transformations applied during the pre-sampling process.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class samplerSettings:

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
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional": {
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",),
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, sampler_name, scheduler, denoise, seed, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
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
                "denoise": denoise,
                "add_noise": "enabled"
            }
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

```
