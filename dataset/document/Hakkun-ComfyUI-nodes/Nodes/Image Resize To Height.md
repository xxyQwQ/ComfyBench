# Image Resize To Height
## Documentation
- Class name: `Image Resize To Height`
- Category: `Hakkun`
- Output node: `False`

This node is designed to resize an image to a specified target height while maintaining its aspect ratio. It calculates the necessary scaling factor based on the target height and the original height of the image, then applies a Lanczos filter for high-quality resizing.
## Input types
### Required
- **`image`**
    - The image to be resized. It is crucial for determining the new dimensions of the image after resizing to the target height.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`target_height`**
    - The desired height of the image after resizing. This parameter directly influences the scaling factor used to adjust the image's height.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resized image with the specified target height, maintaining the original aspect ratio.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageResizeToHeight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (IMAGE_TYPE,),
                "target_height": (INT_TYPE, {"default": 1920, "min": 1, "step": 1}),
            }
        }

    RETURN_TYPES = (IMAGE_TYPE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "calculate"

    CATEGORY = "Hakkun"

    def calculate(self, image, target_height):
        image_size = image.size()
        img_height = int(image_size[1])
        scale_by = target_height/img_height
        return upscale(image, 'lanczos', scale_by)

```
