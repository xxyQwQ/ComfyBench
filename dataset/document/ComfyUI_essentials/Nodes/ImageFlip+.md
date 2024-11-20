---
tags:
- Flip
- Image
- ImageTransformation
---

# 🔧 Image Flip
## Documentation
- Class name: `ImageFlip+`
- Category: `essentials/image manipulation`
- Output node: `False`

The ImageFlip node provides functionality for flipping images along specified axes. It supports flipping images horizontally, vertically, or both, allowing for versatile image manipulation and orientation adjustments.
## Input types
### Required
- **`image`**
    - The image to be flipped. This parameter is crucial for defining the visual content that will undergo the flipping transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`axis`**
    - Specifies the axis or axes along which the image will be flipped. Accepting values 'x', 'y', or 'xy', it determines the direction of the flip, affecting the final appearance of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[str]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The flipped image. This output represents the visual result of applying the specified flip transformation to the input image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageFlip:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "axis": (["x", "y", "xy"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, axis):
        dim = ()
        if "y" in axis:
            dim += (1,)
        if "x" in axis:
            dim += (2,)
        image = torch.flip(image, dim)

        return(image,)

```
