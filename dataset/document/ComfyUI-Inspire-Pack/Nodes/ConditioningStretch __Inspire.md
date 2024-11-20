# Conditioning Stretch (Inspire)
## Documentation
- Class name: `ConditioningStretch __Inspire`
- Category: `InspirePack/conditioning`
- Output node: `False`

The ConditioningStretch node is designed to modify the dimensions of conditioning areas within a given set of conditionings, adjusting them to new specified widths and heights. This node plays a crucial role in adapting conditionings to fit different resolutions or aspect ratios, ensuring that the conditioning effects are appropriately scaled and positioned.
## Input types
### Required
- **`conditioning`**
    - The 'conditioning' input specifies the set of conditionings to be stretched. It is essential for determining the base conditionings that will undergo dimensional adjustments.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`resolutionX`**
    - Specifies the original width of the image or area to which the conditioning was applied. It's crucial for calculating the stretch ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolutionY`**
    - Specifies the original height of the image or area to which the conditioning was applied, essential for calculating the stretch ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`newWidth`**
    - The target width to stretch the conditioning areas to. It defines how much the original conditioning areas will be horizontally expanded or contracted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`newHeight`**
    - The target height to stretch the conditioning areas to. It defines how much the original conditioning areas will be vertically expanded or contracted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Returns the set of conditionings with their areas adjusted to the new specified dimensions, ensuring they are appropriately scaled and positioned.
    - Python dtype: `List[Tuple[str, Dict]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ConditioningStretch:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "resolutionX": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "resolutionY": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "newWidth": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                "newHeight": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                # "scalar": ("INT", {"default": 2, "min": 1, "max": 100, "step": 0.5}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    CATEGORY = "InspirePack/conditioning"

    FUNCTION = 'upscale'

    @staticmethod
    def upscale(conditioning, resolutionX, resolutionY, newWidth, newHeight, scalar=1):
        c = []
        for t in conditioning:

            n = [t[0], t[1].copy()]
            if 'area' in n[1]:
                newWidth *= scalar
                newHeight *= scalar

                x = ((n[1]['area'][3] * 8) * newWidth / resolutionX) // 8
                y = ((n[1]['area'][2] * 8) * newHeight / resolutionY) // 8
                w = ((n[1]['area'][1] * 8) * newWidth / resolutionX) // 8
                h = ((n[1]['area'][0] * 8) * newHeight / resolutionY) // 8

                n[1]['area'] = tuple(map(lambda x: (((int(x) + 7) >> 3) << 3), [h, w, y, x]))

            c.append(n)

        return (c,)

```
