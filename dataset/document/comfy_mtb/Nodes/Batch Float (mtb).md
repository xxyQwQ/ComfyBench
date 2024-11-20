---
tags:
- Batch
- FloatData
---

# Batch Float (mtb)
## Documentation
- Class name: `Batch Float (mtb)`
- Category: `mtb/batch`
- Output node: `False`

Generates a batch of float values with interpolation, offering customization through various parameters such as interpolation mode, count, and range. This node is essential for creating sequences of floats based on specified mathematical easing functions, enabling precise control over the generation of float batches for simulations, animations, or data analysis.
## Input types
### Required
- **`mode`**
    - Specifies the interpolation mode for generating float values, with options like 'Single' or 'Steps' to define how values are interpolated.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`count`**
    - Determines the number of float values to generate, allowing for control over the size of the output batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`min`**
    - Sets the minimum value in the range of generated floats, providing a lower bound for interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max`**
    - Defines the maximum value in the range of generated floats, setting an upper limit for interpolation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`easing`**
    - Selects the easing function to apply during interpolation, such as 'Linear', 'Sine In/Out', or 'Elastic In', affecting the distribution of generated values.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`floats`**
    - Comfy dtype: `FLOATS`
    - The generated batch of interpolated float values, ready for further processing or analysis.
    - Python dtype: `List[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MTB_BatchFloat:
    """Generates a batch of float values with interpolation"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (
                    ["Single", "Steps"],
                    {"default": "Steps"},
                ),
                "count": ("INT", {"default": 2}),
                "min": ("FLOAT", {"default": 0.0, "step": 0.001}),
                "max": ("FLOAT", {"default": 1.0, "step": 0.001}),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In/Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In/Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In/Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In/Out",
                        "Back In",
                        "Back Out",
                        "Back In/Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In/Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In/Out",
                    ],
                    {"default": "Linear"},
                ),
            }
        }

    FUNCTION = "set_floats"
    RETURN_TYPES = ("FLOATS",)
    CATEGORY = "mtb/batch"

    def set_floats(self, mode, count, min, max, easing):
        if mode == "Steps" and count == 1:
            raise ValueError(
                "Steps mode requires at least a count of 2 values"
            )
        keyframes = []
        if mode == "Single":
            keyframes = [min] * count
            return (keyframes,)

        for i in range(count):
            normalized_step = i / (count - 1)
            eased_step = apply_easing(normalized_step, easing)
            eased_value = min + (max - min) * eased_step
            keyframes.append(eased_value)

        return (keyframes,)

```
