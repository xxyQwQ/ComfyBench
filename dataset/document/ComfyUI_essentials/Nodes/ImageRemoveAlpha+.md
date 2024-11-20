---
tags:
- AlphaChannel
- Image
---

# 🔧 Image Remove Alpha
## Documentation
- Class name: `ImageRemoveAlpha+`
- Category: `essentials/image utils`
- Output node: `False`

This node is designed to process images by removing the alpha channel, effectively converting images with transparency into a standard RGB format. It ensures that images are compatible with operations or environments that do not support or require transparency.
## Input types
### Required
- **`image`**
    - The input image with a potential alpha channel. This parameter is crucial for the operation as it determines whether the alpha channel, if present, needs to be removed to convert the image into RGB format.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after the alpha channel has been removed, provided in standard RGB format.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageRemoveAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)

```
