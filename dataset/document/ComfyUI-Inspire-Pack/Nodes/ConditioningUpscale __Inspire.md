---
tags:
- Conditioning
---

# Conditioning Upscale (Inspire)
## Documentation
- Class name: `ConditioningUpscale __Inspire`
- Category: `InspirePack/conditioning`
- Output node: `False`

The ConditioningUpscale node is designed to upscale conditioning areas within a given set of conditionings, adjusting their dimensions based on specified scalar values. It focuses on modifying the 'area' attribute of each conditioning, scaling it to achieve a desired enlargement effect, which is crucial for applications requiring precise control over conditioning dimensions in generative tasks.
## Input types
### Required
- **`conditioning`**
    - The 'conditioning' input represents the set of conditionings to be upscaled. It is essential for determining the areas within the input that will undergo scaling, directly influencing the output's spatial characteristics.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict[str, Any]]]`
- **`scalar`**
    - The 'scalar' input specifies the factor by which the conditioning areas are to be upscaled. It plays a pivotal role in determining the extent of scaling applied to each conditioning area, affecting the overall dimensions of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output is a modified version of the input conditionings, with each 'area' attribute scaled according to the specified scalar value. This is crucial for applications that require adjusted conditioning dimensions for enhanced generative performance.
    - Python dtype: `List[Tuple[str, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningUpscale:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "scalar": ("INT", {"default": 2, "min": 1, "max": 100, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "InspirePack/conditioning"

    FUNCTION = 'upscale'

    @staticmethod
    def upscale(conditioning, scalar):
        c = []
        for t in conditioning:

            n = [t[0], t[1].copy()]
            if 'area' in n[1]:
                n[1]['area'] = tuple(map(lambda x: ((x * scalar + 7) >> 3) << 3, n[1]['area']))

            c.append(n)

        return (c,)

```
