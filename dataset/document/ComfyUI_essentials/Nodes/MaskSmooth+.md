---
tags:
- Blur
- MaskBlur
- VisualEffects
---

# 🔧 Mask Smooth
## Documentation
- Class name: `MaskSmooth+`
- Category: `essentials/mask`
- Output node: `False`

This node applies a Gaussian blur to a given mask to smooth its edges, with the amount of smoothing determined by the input parameter. It's designed to refine mask outlines, making them less jagged and more visually appealing for further processing or application.
## Input types
### Required
- **`mask`**
    - The mask to be smoothed. It represents the area or object of interest that will undergo the smoothing process.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - Determines the intensity of the smoothing effect. A higher value results in more blurring, thus a smoother mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The smoothed version of the input mask, with edges blurred to the specified degree.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskSmooth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 0, "min": 0, "max": 127, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, amount):
        if amount == 0:
            return (mask,)
        
        if amount % 2 == 0:
            amount += 1

        mask = mask > 0.5
        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1).float()

        return (mask,)

```
