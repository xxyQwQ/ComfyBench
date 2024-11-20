# LayerUtility: GradientImage
## Documentation
- Class name: `LayerUtility: GradientImage`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node generates a gradient image based on specified dimensions, angle, and color range. It abstracts the complexity of creating smooth transitions between colors over a defined area, providing a versatile tool for generating backgrounds or layer effects in image processing tasks.
## Input types
### Required
- **`width`**
    - Specifies the width of the gradient image to be generated. It determines the horizontal dimension of the resulting image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the gradient image to be generated. It determines the vertical dimension of the resulting image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - Defines the angle of the gradient direction. This affects how the color transition appears across the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_color`**
    - The starting color of the gradient. It marks the beginning of the color transition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`end_color`**
    - The ending color of the gradient. It marks the end of the color transition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated gradient image as a result of the specified parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GradientImage:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "start_color": ("STRING", {"default": "#FFFFFF"},),
                "end_color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'gradient_image'
    CATEGORY = '😺dzNodes/LayerUtility'

    def gradient_image(self, width, height, angle, start_color, end_color, ):

        ret_image = gradient(start_color, end_color, width, height, angle)

        return (pil2tensor(ret_image), )

```
