# LayerUtility: GradientImage V2
## Documentation
- Class name: `LayerUtility: GradientImage V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to generate gradient images with enhanced features and options compared to its predecessor. It focuses on creating customizable gradient backgrounds or layers by specifying dimensions, colors, and gradient direction, leveraging advanced image processing techniques to achieve more dynamic and visually appealing results.
## Input types
### Required
- **`size`**
    - Specifies the predefined or custom size for the gradient image, allowing for standard or user-defined dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list of str`
- **`custom_width`**
    - Specifies the width of the gradient image when a custom size is selected, affecting the horizontal dimension of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - Specifies the height of the gradient image when a custom size is selected, affecting the vertical dimension of the output image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`angle`**
    - Determines the angle of the gradient, influencing the direction and spread of the color transition in the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_color`**
    - Defines the starting color of the gradient, marking the beginning of the color transition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`end_color`**
    - Defines the ending color of the gradient, marking the end of the color transition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`size_as`**
    - Optionally specifies an image tensor to use its dimensions as the size for the gradient image, overriding 'size', 'custom_width', and 'custom_height' if provided.
    - Comfy dtype: `*`
    - Python dtype: `torch.Tensor or None`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated gradient image, showcasing a smooth transition between the specified start and end colors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GradientImageV2:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        size_list = ['custom']
        size_list.extend(load_custom_size())
        return {
            "required": {
                "size": (size_list,),
                "custom_width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "custom_height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "angle": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "start_color": ("STRING", {"default": "#FFFFFF"},),
                "end_color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'gradient_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def gradient_image_v2(self, size, custom_width, custom_height, angle, start_color, end_color, size_as=None):

        if size_as is not None:
            if size_as.shape[0] > 0:
                _asimage = tensor2pil(size_as[0])
            else:
                _asimage = tensor2pil(size_as)
            width, height = _asimage.size
        else:
            if size == 'custom':
                width = custom_width
                height = custom_height
            else:
                try:
                    _s = size.split('x')
                    width = int(_s[0].strip())
                    height = int(_s[1].strip())
                except Exception as e:
                    log(f"Warning: {NODE_NAME} invalid size, check {custom_size_file}", message_type='warning')
                    width = custom_width
                    height = custom_height


        ret_image = gradient(start_color, end_color, width, height, angle)

        return (pil2tensor(ret_image), )

```
