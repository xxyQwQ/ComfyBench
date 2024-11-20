---
tags:
- ImageSize
- ImageTransformation
---

# 🔧 Get Image Size
## Documentation
- Class name: `GetImageSize+`
- Category: `essentials/image utils`
- Output node: `False`

The GetImageSize node is designed to extract the dimensions of an image, providing essential information such as width, height, and the number of channels or depth. This node is fundamental in image processing tasks where understanding the size and scale of images is crucial for further manipulations or analyses.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image from which the size information is to be extracted. It is crucial for determining the dimensions and depth of the image, which are essential for various image processing applications.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`width`**
    - Comfy dtype: `INT`
    - Represents the width of the input image.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - Represents the height of the input image.
    - Python dtype: `int`
- **`count`**
    - Comfy dtype: `INT`
    - Represents the number of channels or depth of the input image.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [EmptyLatentImage](../../Comfy/Nodes/EmptyLatentImage.md)



## Source code
```python
class GetImageSize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("width", "height", "count")
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image):
        return (image.shape[2], image.shape[1], image.shape[0])

```
