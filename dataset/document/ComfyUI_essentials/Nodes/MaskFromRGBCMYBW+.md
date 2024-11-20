---
tags:
- Mask
- MaskGeneration
---

# 🔧 Mask From RGB/CMY/BW
## Documentation
- Class name: `MaskFromRGBCMYBW+`
- Category: `essentials/mask`
- Output node: `False`

The MaskFromRGBCMYBW node is designed to generate masks based on specific RGB, CMY, and BW (black and white) thresholds applied to an input image. It allows for the creation of multiple masks corresponding to the primary and secondary colors, as well as black and white, based on the defined thresholds, enabling precise control over color-based segmentation for various image processing tasks.
## Input types
### Required
- **`image`**
    - The input image to be processed for mask generation based on RGB, CMY, and BW thresholds.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`threshold_r`**
    - The threshold value for the red component, used to determine the boundaries for red, cyan, magenta, and black/white masks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`threshold_g`**
    - The threshold value for the green component, used to determine the boundaries for green, cyan, yellow, and black/white masks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`threshold_b`**
    - The threshold value for the blue component, used to determine the boundaries for blue, magenta, yellow, and black/white masks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`red`**
    - Comfy dtype: `MASK`
    - The mask generated for the red color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`green`**
    - Comfy dtype: `MASK`
    - The mask generated for the green color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`blue`**
    - Comfy dtype: `MASK`
    - The mask generated for the blue color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`cyan`**
    - Comfy dtype: `MASK`
    - The mask generated for the cyan color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`magenta`**
    - Comfy dtype: `MASK`
    - The mask generated for the magenta color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`yellow`**
    - Comfy dtype: `MASK`
    - The mask generated for the yellow color based on the threshold.
    - Python dtype: `torch.Tensor`
- **`black`**
    - Comfy dtype: `MASK`
    - The mask generated for black based on the threshold.
    - Python dtype: `torch.Tensor`
- **`white`**
    - Comfy dtype: `MASK`
    - The mask generated for white based on the threshold.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskFromRGBCMYBW:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)

```
