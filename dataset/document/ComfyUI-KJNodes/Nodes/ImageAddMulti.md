---
tags:
- Image
- ImageBlend
- ImageComposite
---

# Image Add Multi
## Documentation
- Class name: `ImageAddMulti`
- Category: `KJNodes/image`
- Output node: `False`

This node blends multiple images together using various blending modes such as add, subtract, multiply, and difference. It allows for dynamic input count adjustment, enabling the combination of a flexible number of images into a single output image.
## Input types
### Required
- **`inputcount`**
    - Specifies the number of images to blend together. It determines how many image inputs the node will process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_i`**
    - Represents an image to be blended. The index 'i' is dynamic, starting from 1 and increasing based on the 'inputcount'. This allows for a flexible number of images to be processed and blended together.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`blending`**
    - Defines the blending mode to be used for combining the images. Supported modes are add, subtract, multiply, and difference.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting image after blending the input images according to the specified blending mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageAddMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
                "blending": (
                [   'add',
                    'subtract',
                    'multiply',
                    'difference',
                ],
                {
                "default": 'add'
                }),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "add"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Add blends multiple images together.    
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def add(self, inputcount, blending, **kwargs):
        image = kwargs["image_1"]
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            if blending == "add":
                image = torch.add(image * 0.5, new_image * 0.5)
            elif blending == "subtract":
                image = torch.sub(image * 0.5, new_image * 0.5)
            elif blending == "multiply":
                image = torch.mul(image * 0.5, new_image * 0.5)
            elif blending == "difference":
                image = torch.sub(image, new_image)
        return (image,)    

```
