---
tags:
- Sampling
---

# PreSampling (DynamicCFG)
## Documentation
- Class name: `easy preSamplingDynamicCFG`
- Category: `EasyUse/PreSampling`
- Output node: `True`

The node 'easy preSamplingDynamicCFG' is designed to configure and apply dynamic control flow guidance (CFG) settings for pre-sampling processes in generative models. It allows for the customization of CFG parameters to influence the generation process, aiming to enhance the quality and relevance of the generated content by dynamically adjusting the guidance based on specified conditions.
## Input types
### Required
- **`pipe`**
    - Specifies the pipeline configuration to which the dynamic CFG settings will be applied. It is essential for integrating the dynamic CFG adjustments into the pre-sampling process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict[str, Any]`
- **`steps`**
    - Determines the number of steps for the pre-sampling process, affecting the granularity and detail of the generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Sets the base CFG scale, influencing the strength of the conditioning on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cfg_mode`**
    - Defines the mode of CFG application, allowing for different strategies in guiding the generative model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `DynThresh.Modes`
- **`cfg_scale_min`**
    - Specifies the minimum CFG scale, setting a lower bound on the guidance strength.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the sampler to be used in conjunction with the dynamic CFG settings, impacting the method of generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`denoise`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`image_to_latent`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`latent`**
    - unknown
    - Comfy dtype: `LATENT`
    - Python dtype: `unknown`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the updated pipeline configuration, now including the dynamic CFG settings for pre-sampling.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class dynamicCFGSettings:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"pipe": ("PIPE_LINE",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "cfg_mode": (DynThresh.Modes,),
                     "cfg_scale_min": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.5}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS+new_schedulers,),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED_NUM}),
                     },
                "optional":{
                    "image_to_latent": ("IMAGE",),
                    "latent": ("LATENT",)
                },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_NODE = True

    FUNCTION = "settings"
    CATEGORY = "EasyUse/PreSampling"

    def settings(self, pipe, steps, cfg, cfg_mode, cfg_scale_min,sampler_name, scheduler, denoise, seed, image_to_latent=None, latent=None, prompt=None, extra_pnginfo=None, my_unique_id=None):


        dynamic_thresh = DynThresh(7.0, 1.0,"CONSTANT", 0, cfg_mode, cfg_scale_min, 0, 0, 999, False,
                                   "MEAN", "AD", 1)

        def sampler_dyn_thresh(args):
            input = args["input"]
            cond = input - args["cond"]
            uncond = input - args["uncond"]
            cond_scale = args["cond_scale"]
            time_step = args["timestep"]
            dynamic_thresh.step = 999 - time_step[0]

            return input - dynamic_thresh.dynthresh(cond, uncond, cond_scale, None)

        model = pipe['model']

        m = model.clone()
        m.set_model_sampler_cfg_function(sampler_dyn_thresh)

        # 图生图转换
        vae = pipe["vae"]
        batch_size = pipe["loader_settings"]["batch_size"] if "batch_size" in pipe["loader_settings"] else 1
        if image_to_latent is not None:
            samples = {"samples": vae.encode(image_to_latent[:, :, :, :3])}
            samples = RepeatLatentBatch().repeat(samples, batch_size)[0]
            images = image_to_latent
        elif latent is not None:
            samples = RepeatLatentBatch().repeat(latent, batch_size)[0]
            images = pipe["images"]
        else:
            samples = pipe["samples"]
            images = pipe["images"]

        new_pipe = {
            "model": m,
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
                "denoise": denoise
            },
        }

        del pipe

        return {"ui": {"value": [seed]}, "result": (new_pipe,)}

```
