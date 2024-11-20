---
tags:
- RegionalPrompt
---

# Regional Conditioning Simple (Inspire)
## Documentation
- Class name: `RegionalConditioningSimple __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in applying regional conditioning to an input based on a specified mask, strength, and textual prompt. It leverages CLIP text encoding and mask-based conditioning to enhance or modify specific regions of an input according to the given prompt, allowing for targeted adjustments or enhancements.
## Input types
### Required
- **`clip`**
    - The CLIP model to be used for text encoding, which plays a crucial role in interpreting the prompt and applying the corresponding conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`mask`**
    - A mask that defines the region to apply the conditioning to, determining where the effects of the prompt will be localized.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - Defines the intensity of the applied conditioning, allowing for fine-tuning how strongly the prompt influences the specified region.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Specifies how the conditioning area is determined, either using default settings or based on mask bounds, affecting the scope of the applied conditioning.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`prompt`**
    - The textual prompt that guides the conditioning process, directly influencing the nature of the applied adjustments.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The result of the conditioning process, tailored to the specified region, strength, and prompt, ready for further processing or application.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalConditioningSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "mask": ("MASK",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(clip, mask, strength, set_cond_area, prompt):
        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return (conditioning, )

```
