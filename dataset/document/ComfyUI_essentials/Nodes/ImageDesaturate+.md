---
tags:
- ImageEnhancement
- ImageTransformation
- VisualEffects
---

# 🔧 Image Desaturate
## Documentation
- Class name: `ImageDesaturate+`
- Category: `essentials/image processing`
- Output node: `False`

The ImageDesaturate node is designed for adjusting the saturation level of an image, allowing for partial to full desaturation based on a specified method and factor. It supports different methods of calculating grayscale values, enabling nuanced control over the desaturation process.
## Input types
### Required
- **`image`**
    - The input image to be desaturated. It serves as the primary data for the desaturation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`factor`**
    - A value between 0 and 1 that determines the intensity of the desaturation effect. A factor of 1 results in full desaturation, while 0 leaves the image unchanged.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`method`**
    - The method used to calculate the grayscale values during desaturation. Options include luminance (Rec.709), luminance (Rec.601), average, and lightness, each affecting the output image's appearance differently.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The desaturated image, with its saturation level adjusted according to the specified factor and method.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageDesaturate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "factor": ("FLOAT", { "default": 1.00, "min": 0.00, "max": 1.00, "step": 0.05, }),
                "method": (["luminance (Rec.709)", "luminance (Rec.601)", "average", "lightness"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image processing"

    def execute(self, image, factor, method):
        if method == "luminance (Rec.709)":
            grayscale = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        elif method == "luminance (Rec.601)":
            grayscale = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        elif method == "average":
            grayscale = image.mean(dim=3)
        elif method == "lightness":
            grayscale = (torch.max(image, dim=3)[0] + torch.min(image, dim=3)[0]) / 2

        grayscale = (1.0 - factor) * image + factor * grayscale.unsqueeze(-1).repeat(1, 1, 1, 3)
        grayscale = torch.clamp(grayscale, 0, 1)

        return(grayscale,)

```
