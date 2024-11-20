---
tags:
- Mask
- MaskGeneration
---

# 🔧 Mask From Color
## Documentation
- Class name: `MaskFromColor+`
- Category: `essentials/mask`
- Output node: `False`

This node generates a mask from an image based on specified color values and a threshold. It allows for the selective isolation of parts of an image that match the given color within a certain tolerance, enabling targeted image manipulation or analysis.
## Input types
### Required
- **`image`**
    - The input image from which the mask will be generated. It serves as the basis for identifying areas that match the specified color criteria.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`red`**
    - The red component of the target color. It defines the red value that the node will use to generate the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`green`**
    - The green component of the target color. It defines the green value that the node will use to generate the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blue`**
    - The blue component of the target color. It defines the blue value that the node will use to generate the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`threshold`**
    - The tolerance for color matching. It specifies how closely the colors in the image must match the target color to be included in the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output mask indicating areas of the image that match the specified color within the given threshold.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskFromColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "red": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "green": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "blue": ("INT", { "default": 255, "min": 0, "max": 255, "step": 1, }),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, image, red, green, blue, threshold):
        temp = (torch.clamp(image, 0, 1.0) * 255.0).round().to(torch.int)
        color = torch.tensor([red, green, blue])
        lower_bound = (color - threshold).clamp(min=0)
        upper_bound = (color + threshold).clamp(max=255)
        lower_bound = lower_bound.view(1, 1, 1, 3)
        upper_bound = upper_bound.view(1, 1, 1, 3)
        mask = (temp >= lower_bound) & (temp <= upper_bound)
        mask = mask.all(dim=-1)
        mask = mask.float()

        return (mask, )

```
