---
tags:
- AlphaChannel
- Image
---

# AddAlpha
## Documentation
- Class name: `AddAlpha`
- Category: `Bmad/image`
- Output node: `False`

The `AddAlpha` node is designed to add an alpha channel to RGB images, allowing for the manipulation of image transparency. It supports optional inversion of the alpha channel, providing flexibility in handling transparency effects.
## Input types
### Required
- **`rgb_image`**
    - The RGB image to which an alpha channel will be added. This is the primary image input for transparency manipulation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`alpha`**
    - An optional alpha channel to be added to the RGB image. If provided, it specifies the transparency levels for the image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`method`**
    - An optional method specifying how the alpha channel should be applied. It can either add the alpha channel directly or invert it, affecting the transparency effect.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image with an alpha channel added, enabling transparency manipulation.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AddAlpha:
    method = ["default", "invert"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rgb_image": ("IMAGE",),
            },
            "optional": {
                "alpha": ("IMAGE",),
                "method": (cls.method, {"default": cls.method[0]}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_alpha"
    CATEGORY = images_category_path

    def add_alpha(self, rgb_image, alpha=None, method=None):
        rgb_image = tensor2opencv(rgb_image, 3)
        rgba = cv.cvtColor(rgb_image, cv.COLOR_RGB2RGBA)
        if alpha is not None:
            alpha = tensor2opencv(alpha, 1)
            rgba[:, :, 3] = alpha if method == self.method[0] else 255 - alpha
        rgba = opencv2tensor(rgba)
        return (rgba,)

```
