---
tags:
- SamplerScheduler
- Sampling
---

# Impact Scheduler Adapter
## Documentation
- Class name: `ImpactSchedulerAdapter`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactSchedulerAdapter node is designed to adapt and select scheduling strategies for various tasks, allowing for dynamic adjustment of scheduling based on specific conditions or preferences.
## Input types
### Required
- **`scheduler`**
    - Specifies the primary scheduler to be used, with an option to default to a pre-defined input if no specific scheduler is provided.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `comfy.samplers.KSampler.SCHEDULERS`
- **`extra_scheduler`**
    - Allows for the specification of an additional scheduler, offering options such as 'None', 'AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]', to override the primary scheduler if needed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`scheduler`**
    - Comfy dtype: `COMBO[STRING]`
    - Outputs the selected scheduler, which could either be the primary scheduler or an overridden scheduler specified by the 'extra_scheduler' input.
    - Python dtype: `Tuple[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImpactSchedulerAdapter:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"defaultInput": True, }),
            "extra_scheduler": (['None', 'AYS SDXL', 'AYS SD1', 'AYS SVD', 'GITS[coeff=1.2]'],),
        }}

    CATEGORY = "ImpactPack/Util"

    RETURN_TYPES = (core.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)

    FUNCTION = "doit"

    def doit(self, scheduler, extra_scheduler):
        if extra_scheduler != 'None':
            return (extra_scheduler,)

        return (scheduler,)

```
