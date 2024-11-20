---
tags:
- SamplerScheduler
- Sampling
---

# XY Input: Sampler/Scheduler
## Documentation
- Class name: `XY Input: Sampler_Scheduler`
- Category: `Efficiency Nodes/XY Inputs`
- Output node: `False`

This node is designed to process and validate combinations of samplers and schedulers for use in sampling operations. It ensures that the provided sampler and scheduler names are valid within the system's defined bounds, potentially adjusting or providing feedback if invalid options are encountered. The node's functionality is crucial for configuring and optimizing sampling processes, facilitating the selection of appropriate sampling and scheduling strategies based on user input or system requirements.
## Input types
### Required
- **`target_parameter`**
    - Specifies the target parameter for optimization, such as 'sampler', 'scheduler', or a combination of both. It determines the focus of the node's processing and directly influences the selection and validation process, guiding the optimization of sampling operations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`input_count`**
    - Defines the number of inputs to be considered for the target parameter. This count affects the node's execution by determining how many sampler or scheduler options are evaluated, influencing the comprehensiveness of the validation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sampler_i`**
    - Represents one of potentially multiple sampler names to be validated. The index 'i' varies, indicating each sampler's position in the sequence for evaluation. This parameter is crucial for assessing the suitability of various samplers within the specified sampling strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler_i`**
    - Denotes one of potentially multiple scheduler names to be validated, with 'i' indicating the scheduler's position in the sequence. It plays a key role in evaluating the compatibility and efficiency of different scheduling algorithms for the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`X or Y`**
    - Comfy dtype: `XY`
    - This output represents the result of the validation process, either confirming the suitability of a sampler or scheduler ('X') or providing an alternative recommendation ('Y'). It ensures that the chosen strategies are optimal for the intended sampling operation.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class TSC_XYplot_Sampler_Scheduler:
    parameters = ["sampler", "scheduler", "sampler & scheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + SCHEDULERS
        inputs = {
            "required": {
                "target_parameter": (cls.parameters,),
                "input_count": ("INT", {"default": XYPLOT_DEF, "min": 0, "max": XYPLOT_LIM, "step": 1})
            }
        }
        for i in range(1, XYPLOT_LIM+1):
            inputs["required"][f"sampler_{i}"] = (samplers,)
            inputs["required"][f"scheduler_{i}"] = (schedulers,)

        return inputs

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Inputs"

    def xy_value(self, target_parameter, input_count, **kwargs):
        if target_parameter == "scheduler":
            xy_type = "Scheduler"
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            xy_value = [scheduler for scheduler in schedulers if scheduler != "None"]
        elif target_parameter == "sampler":
            xy_type = "Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            xy_value = [(sampler, None) for sampler in samplers if sampler != "None"]
        else:
            xy_type = "Sampler"
            samplers = [kwargs.get(f"sampler_{i}") for i in range(1, input_count + 1)]
            schedulers = [kwargs.get(f"scheduler_{i}") for i in range(1, input_count + 1)]
            xy_value = [(sampler, scheduler if scheduler != "None" else None) for sampler,
            scheduler in zip(samplers, schedulers) if sampler != "None"]

        return ((xy_type, xy_value),) if xy_value else (None,)

```
