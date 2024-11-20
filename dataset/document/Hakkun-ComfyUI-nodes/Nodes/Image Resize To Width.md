---
tags:
- ImageResize
- ImageScaling
- ImageSize
- ImageTransformation
---

# Image Resize To Width
## Documentation
- Class name: `Image Resize To Width`
- Category: `Hakkun`
- Output node: `False`

The node is designed to resize an image to a specific width while maintaining its aspect ratio, using a specified upscale method to ensure quality preservation.
## Input types
### Required
- **`image`**
    - The image to be resized. This parameter is crucial as it provides the source image that will undergo resizing to match the target width.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target_width`**
    - The target width to which the image will be resized. This parameter determines the scale of the resizing operation and directly affects the output image's dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resized image with the specified target width, maintaining the original aspect ratio.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageResizeToWidth:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_width": (INT_TYPE, {"default": 1920, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (IMAGE_TYPE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, target_width):
        image_size = image.size()
        img_width = int(image_size[2])
        scale_by = target_width/img_width
        return upscale(image, 'lanczos', scale_by)

```
