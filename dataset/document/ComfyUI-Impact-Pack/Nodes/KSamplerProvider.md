---
tags:
- SamplerScheduler
- Sampling
---

# KSamplerProvider
## Documentation
- Class name: `KSamplerProvider`
- Category: `ImpactPack/Sampler`
- Output node: `False`

The KSamplerProvider node is designed to offer a flexible and efficient way to provide sampling functionality within a generative model framework. It abstracts the complexity of sampling algorithms, allowing for easy integration and customization of sampling strategies for various applications.
## Input types
### Required
- **`seed`**
    - Random seed to use for generating CPU noise for sampling, influencing the randomness of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Specifies the total number of sampling steps to be executed, affecting the granularity of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Classifier free guidance value, used to control the influence of the conditioning on the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampler to be used, allowing for the selection of different sampling algorithms.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the noise schedule to be applied during the sampling process, impacting the progression of noise reduction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Determines the amount of noise to remove at the start of the sampling process, affecting the initial state of the input latent.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`basic_pipe`**
    - Serves as the primary input for the sampling process, encapsulating essential components like the model and conditioning parameters.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `tuple`
### Optional
- **`scheduler_func_opt`**
    - An optional parameter that allows specifying a custom noise schedule generation function, offering further customization of the sampling process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `Optional[Callable]`
## Output types
- **`ksampler`**
    - Comfy dtype: `KSAMPLER`
    - The resulting sampler object, configured and ready for use in generating regional prompts or other sampling-based tasks.
    - Python dtype: `KSamplerWrapper`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (core.SCHEDULERS, ),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "basic_pipe": "basic_pipe input for sampling",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=scheduler_func_opt)
        return (sampler, )

```
