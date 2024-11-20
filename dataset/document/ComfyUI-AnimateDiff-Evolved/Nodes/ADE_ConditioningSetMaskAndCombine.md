---
tags:
- Conditioning
---

# Set Props and Combine Cond 🎭🅐🅓
## Documentation
- Class name: `ADE_ConditioningSetMaskAndCombine`
- Category: `Animate Diff 🎭🅐🅓/conditioning/single cond ops`
- Output node: `False`

This node specializes in the advanced manipulation of conditioning data for generative models, specifically focusing on the application of masks, the combination of multiple conditioning inputs, and the adjustment of their influence through strength parameters. It enables the dynamic alteration and enhancement of conditioning inputs to tailor the generative process more precisely to desired outcomes.
## Input types
### Required
- **`cond`**
    - The original conditioning data to be modified or enhanced.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `Tensor`
- **`cond_ADD`**
    - Additional conditioning data to be combined with the original conditioning, allowing for the introduction of new elements or modifications.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `Tensor`
- **`strength`**
    - A scalar value that determines the intensity of the applied modifications or enhancements on the conditioning data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Specifies the area within the conditioning data that the modifications, such as masking or strength adjustments, should be applied to.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`opt_mask`**
    - An optional mask that can be applied to the conditioning data to selectively modify or enhance specific regions.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`opt_lora_hook`**
    - An optional hook for applying LoRA adjustments to the conditioning data, enabling more fine-grained control over the modifications.
    - Comfy dtype: `LORA_HOOK`
    - Python dtype: `LoraHookGroup`
- **`opt_timesteps`**
    - Optional timesteps conditioning that allows for temporal adjustments to the conditioning data, useful in sequential generative processes.
    - Comfy dtype: `TIMESTEPS_COND`
    - Python dtype: `TimestepsCond`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting conditioning data after applying the specified modifications, combinations, and enhancements.
    - Python dtype: `Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningSetMaskAndCombineHooked:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cond": ("CONDITIONING",),
                "cond_ADD": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (COND_CONST._LIST_COND_AREA,),
            },
            "optional": {
                "opt_mask": ("MASK", ),
                "opt_lora_hook": ("LORA_HOOK",),
                "opt_timesteps": ("TIMESTEPS_COND",)
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "Animate Diff 🎭🅐🅓/conditioning/single cond ops"
    FUNCTION = "append_and_combine"

    def append_and_combine(self, cond, cond_ADD,
                           strength: float, set_cond_area: str,
                           opt_mask: Tensor=None, opt_lora_hook: LoraHookGroup=None, opt_timesteps: TimestepsCond=None):
        (final_conditioning,) = set_mask_and_combine_conds(conds=[cond], new_conds=[cond_ADD],
                                                                    strength=strength, set_cond_area=set_cond_area,
                                                                    opt_mask=opt_mask, opt_lora_hook=opt_lora_hook, opt_timesteps=opt_timesteps)
        return (final_conditioning,)

```
