---
tags:
- Noise
---

# Composite Noise (Inspire)
## Documentation
- Class name: `CompositeNoise __Inspire`
- Category: `InspirePack/Prompt`
- Output node: `False`

The CompositeNoise node in the Inspire pack is designed to enhance the generation process by applying a composite noise strategy. It integrates various noise manipulation techniques to adjust and refine the generated outputs, aiming to improve the quality and diversity of the results. This node likely combines different types of noise and processing methods to achieve a more nuanced and controlled generation outcome, making it a versatile tool for creative and generative tasks.
## Input types
### Required
- **`destination`**
    - Specifies the destination for the noise application, indicating where the noise will be integrated within the generation process.
    - Comfy dtype: `NOISE`
    - Python dtype: `str`
- **`source`**
    - Identifies the source of the noise, which could be an external input or generated within the system, providing the basis for noise manipulation.
    - Comfy dtype: `NOISE`
    - Python dtype: `str`
- **`mode`**
    - Determines the mode of noise application, such as additive or multiplicative, affecting how the noise interacts with the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x`**
    - Represents a specific parameter or variable related to the noise application, potentially influencing the direction or intensity of the noise.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - Another parameter or variable that may affect the noise application process, offering additional control over the manipulation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`noise`**
    - Comfy dtype: `NOISE`
    - The generated or modified noise after application, showcasing the effects of the composite noise manipulation strategies.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CompositeNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "destination": ("NOISE",),
                "source": ("NOISE",),
                "mode": (["center", "left-top", "right-top", "left-bottom", "right-bottom", "xy"], ),
                "x": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                "y": ("INT", {"default": 0, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
            },
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    def doit(self, destination, source, mode, x, y):
        new_tensor = destination.clone()

        if mode == 'center':
            y1 = (new_tensor.size(2) - source.size(2)) // 2
            x1 = (new_tensor.size(3) - source.size(3)) // 2
        elif mode == 'left-top':
            y1 = 0
            x1 = 0
        elif mode == 'right-top':
            y1 = 0
            x1 = new_tensor.size(2) - source.size(2)
        elif mode == 'left-bottom':
            y1 = new_tensor.size(3) - source.size(3)
            x1 = 0
        elif mode == 'right-bottom':
            y1 = new_tensor.size(3) - source.size(3)
            x1 = new_tensor.size(2) - source.size(2)
        else:  # mode == 'xy':
            x1 = max(0, x)
            y1 = max(0, y)

        # raw coordinates
        y2 = y1 + source.size(2)
        x2 = x1 + source.size(3)

        # bounding for destination
        top = max(0, y1)
        left = max(0, x1)
        bottom = min(new_tensor.size(2), y2)
        right = min(new_tensor.size(3), x2)

        # bounding for source
        left_gap = left - x1
        top_gap = top - y1

        width = right - left
        height = bottom - top

        height = min(height, y1 + source.size(2) - top)
        width = min(width, x1 + source.size(3) - left)

        # composite
        new_tensor[:, :, top:top + height, left:left + width] = source[:, :, top_gap:top_gap + height, left_gap:left_gap + width]

        return (new_tensor,)

```
