---
tags:
- Image
---

# Image Concatenate Multi
## Documentation
- Class name: `ImageConcatMulti`
- Category: `KJNodes/image`
- Output node: `False`

The ImageConcatMulti node is designed to concatenate multiple images into a single image. It allows for the dynamic specification of the number of input images, the direction of concatenation, and whether to match the sizes of the images being concatenated. This functionality is useful for creating composite images or arranging multiple images in a specific layout.
## Input types
### Required
- **`inputcount`**
    - Specifies the number of images to concatenate. This allows for dynamic adjustment of the node's inputs based on user requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_i`**
    - Represents any of the images to be concatenated, allowing for a flexible number of image inputs beyond the first two specified. This accommodates the dynamic input count specified by the user.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`direction`**
    - Determines the direction in which the images will be concatenated (right, down, left, up). This affects the final layout of the concatenated image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`match_image_size`**
    - Indicates whether the sizes of the images should be matched before concatenation. This ensures uniformity in the dimensions of the resulting image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting image after concatenation of the input images. This composite image reflects the specified direction and size matching settings.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageConcatMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "direction": (
                [   'right',
                    'down',
                    'left',
                    'up',
                ],
            {
            "default": 'right'
             }),
            "match_image_size": ("BOOLEAN", {"default": False}),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image from multiple images.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, direction, match_image_size, **kwargs):
        image = kwargs["image_1"]
        first_image_shape = None
        if first_image_shape is None:
            first_image_shape = image.shape
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            image, = ImageConcanate.concanate(self, image, new_image, direction, match_image_size, first_image_shape=first_image_shape)
        first_image_shape = None
        return (image,)

```
