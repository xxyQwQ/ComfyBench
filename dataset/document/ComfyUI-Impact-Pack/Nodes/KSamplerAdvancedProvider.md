---
tags:
- SamplerScheduler
- Sampling
---

# KSamplerAdvancedProvider
## Documentation
- Class name: `KSamplerAdvancedProvider`
- Category: `ImpactPack/Sampler`
- Output node: `False`

The KSamplerAdvancedProvider node is designed to offer advanced sampling capabilities within a generative modeling framework. It likely extends or enhances the functionality of basic sampling by providing more sophisticated methods or options for sampling, tailored to specific needs or higher complexity scenarios.
## Input types
### Required
- **`cfg`**
    - Specifies the classifier free guidance value, which influences the direction and strength of the generative model's output, guiding it towards desired characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Determines the specific sampling algorithm to be used, allowing for customization of the sampling process based on the model's requirements or desired outcomes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Defines the noise schedule for the sampling process, impacting how noise is added or removed during generation to control the quality and characteristics of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sigma_factor`**
    - A multiplier for the noise schedule, adjusting the level of noise applied in the sampling process, thereby fine-tuning the output's fidelity or diversity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`basic_pipe`**
    - Represents the basic pipeline input for sampling, serving as the foundational data or model configuration upon which the sampling process operates.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `str`
### Optional
- **`sampler_opt`**
    - [OPTIONAL] Allows for the use of an externally defined sampler instead of the node's internal sampler, offering flexibility in the sampling approach.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `str`
- **`scheduler_func_opt`**
    - [OPTIONAL] Specifies a custom function for generating the noise schedule, providing an alternative to the predefined scheduler options for more tailored sampling dynamics.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`ksampler_advanced`**
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Produces an advanced sampler configuration, capable of being utilized in complex generative tasks that require nuanced control over the sampling process.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class KSamplerAdvancedProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                                "scheduler": (core.SCHEDULERS, ),
                                "sigma_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                                "basic_pipe": ("BASIC_PIPE", )
                             },
                "optional": {
                                "sampler_opt": ("SAMPLER", ),
                                "scheduler_func_opt": ("SCHEDULER_FUNC",),
                            }
                }

    TOOLTIPS = {
        "input": {
            "cfg": "classifier free guidance value",
            "sampler_name": "sampler",
            "scheduler": "noise schedule",
            "sigma_factor": "Multiplier of noise schedule",
            "basic_pipe": "basic_pipe input for sampling",
            "sampler_opt": "[OPTIONAL] Uses the passed sampler instead of internal impact_sampler.",
            "scheduler_func_opt": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored.",
        },
        "output": ("sampler wrapper. (Can be used when generating a regional_prompt.)", )
    }

    RETURN_TYPES = ("KSAMPLER_ADVANCED",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(cfg, sampler_name, scheduler, basic_pipe, sigma_factor=1.0, sampler_opt=None, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerAdvancedWrapper(model, cfg, sampler_name, scheduler, positive, negative, sampler_opt=sampler_opt, sigma_factor=sigma_factor, scheduler_func=scheduler_func_opt)
        return (sampler, )

```
