---
tags:
- Color
---

# LayerUtility: ColorImage
## Documentation
- Class name: `LayerUtility: ColorImage`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ColorImage node generates a solid color image based on specified dimensions and color. It serves as a utility for creating background layers or color fills in image processing workflows.
## Input types
### Required
- **`width`**
    - Specifies the width of the generated image. It determines how wide the image will be.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the generated image. It determines the height of the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - Defines the color of the generated image. The color is specified in a string format, typically as a hex code.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a solid color image with the specified dimensions and color.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorImage:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'color_image'
    CATEGORY = '😺dzNodes/LayerUtility'

    def color_image(self, width, height, color, ):

        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), )

```
