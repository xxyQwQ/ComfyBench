# SamplerEulerCFG++
## Documentation
- Class name: `SamplerEulerCFGpp`
- Category: `_for_testing`
- Output node: `False`

The SamplerEulerCFGpp node provides a mechanism to select between two versions of a sampler based on the Euler method for generating samples, offering a choice between a regular and an alternative configuration.
## Input types
### Required
- **`version`**
    - Determines the version of the Euler method-based sampler to be used, allowing selection between a 'regular' and an 'alternative' configuration. This choice affects the underlying sampling algorithm and its behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`sampler`**
    - Comfy dtype: `SAMPLER`
    - Outputs a sampler configured according to the selected version, capable of generating samples using the Euler method.
    - Python dtype: `comfy.samplers.KSampler`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SamplerEulerCFGpp:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"version": (["regular", "alternative"],),}
               }
    RETURN_TYPES = ("SAMPLER",)
    # CATEGORY = "sampling/custom_sampling/samplers"
    CATEGORY = "_for_testing"

    FUNCTION = "get_sampler"

    def get_sampler(self, version):
        if version == "alternative":
            sampler = comfy.samplers.KSAMPLER(sample_euler_pp)
        else:
            sampler = comfy.samplers.ksampler("euler_cfg_pp")
        return (sampler, )

```
