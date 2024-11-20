---
tags:
- SamplerScheduler
- Sampling
---

# XY Inputs: Sampler/Scheduler //EasyUse
## Documentation
- Class name: `easy XYInputs: Sampler_Scheduler`
- Category: `EasyUse/XY Inputs`
- Output node: `False`

This node is designed to integrate and manage the interaction between a sampler and a scheduler within a sampling process. It facilitates the configuration and execution of sampling strategies by allowing users to specify a sampler and a scheduler, along with their respective parameters, to generate or modify sample outputs effectively.
## Input types
### Required
- **`target_parameter`**
    - This parameter is crucial for defining the specific target or goal that the sampling process aims to achieve, directly influencing the outcome and effectiveness of the sampling strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`input_count`**
    - Determines the number of inputs to be processed, affecting the node's execution by scaling the sampling operation according to the provided input volume.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sampler_i`**
    - Specifies the index of the sampler to be used, selecting from a predefined list of available samplers. This choice impacts the sampling method and its characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`scheduler_i`**
    - Indicates the index of the scheduler, chosen from a set of available scheduling algorithms. This selection influences the timing and adjustment of sampling parameters throughout the process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
## Output types
- **`X or Y`**
    - Comfy dtype: `X_Y`
    - Outputs a result that is contingent upon the configuration and interaction of the sampler and scheduler. The exact nature of the output (X or Y) depends on the specific parameters and algorithms employed.
    - Python dtype: `torch.FloatTensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class XYplot_Sampler_Scheduler:
    parameters = ["sampler", "scheduler", "sampler & scheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS
        inputs = {
            "required": {
                "target_parameter": (cls.parameters,),
                "input_count": ("INT", {"default": 1, "min": 1, "max": 30, "step": 1})
            }
        }
        for i in range(1, 30 + 1):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)

        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, target_parameter, input_count, **kwargs):
        axis, values, = None, None,
        if target_parameter == "scheduler":
            axis = "advanced: Scheduler"
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = [scheduler for scheduler in schedulers if scheduler != "None"]
        elif target_parameter == "sampler":
            axis = "advanced: Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            values = [sampler for sampler in samplers if sampler != "None"]
        else:
            axis = "advanced: Sampler&Scheduler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            values = []
            for sampler, scheduler in zip(samplers, schedulers):
                sampler = sampler if sampler else 'None'
                scheduler = scheduler if scheduler else 'None'
                values.append(sampler +','+ scheduler)
        values = "; ".join(values)
        return ({"axis": axis, "values": values},) if values else (None,)

```
