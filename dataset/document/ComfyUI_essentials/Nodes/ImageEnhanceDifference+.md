---
tags:
- Comparison
---

# 🔧 Image Enhance Difference
## Documentation
- Class name: `ImageEnhanceDifference+`
- Category: `essentials/image analysis`
- Output node: `False`

This node is designed to enhance and highlight the differences between two images by applying a power transformation. It is useful for visualizing changes or discrepancies between two images in a more pronounced manner.
## Input types
### Required
- **`image1`**
    - The first image to compare. It serves as the baseline for the comparison.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to compare against the first. This image is adjusted to match the dimensions of the first image if necessary.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`exponent`**
    - A factor that controls the intensity of the enhancement. Higher values increase the contrast of the differences.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced difference image, highlighting discrepancies between the input images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageEnhanceDifference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "exponent": ("FLOAT", { "default": 0.75, "min": 0.00, "max": 1.00, "step": 0.05, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image analysis"

    def execute(self, image1, image2, exponent):
        if image1.shape[1:] != image2.shape[1:]:
            image2 = comfy.utils.common_upscale(image2.permute([0,3,1,2]), image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center').permute([0,2,3,1])

        diff_image = image1 - image2
        diff_image = torch.pow(diff_image, exponent)
        diff_image = torch.clamp(diff_image, 0, 1)

        return(diff_image,)

```
