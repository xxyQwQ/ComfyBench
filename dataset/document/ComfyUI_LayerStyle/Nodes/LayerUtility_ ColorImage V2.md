---
tags:
- Color
---

# LayerUtility: ColorImage V2
## Documentation
- Class name: `LayerUtility: ColorImage V2`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to generate a solid color image based on specified dimensions and color. It abstracts the process of creating an image filled with a single color, facilitating the creation of background layers or color-based placeholders in graphic projects.
## Input types
### Required
- **`size`**
    - Determines the size of the generated image, allowing for predefined or custom dimensions. It enables flexibility in defining the image's overall dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`custom_width`**
    - Specifies the custom width for the image when 'size' is set to 'custom'. It defines the width of the image in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_height`**
    - Specifies the custom height for the image when 'size' is set to 'custom'. It defines the height of the image in pixels.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`color`**
    - Defines the color of the generated image. This parameter allows for customization of the image's appearance, enabling the creation of images with any desired solid color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`size_as`**
    - Optionally specifies an image to match its size. If provided, the generated image will match the dimensions of this reference image.
    - Comfy dtype: `*`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a solid color image with the specified dimensions and color. It serves as a versatile element for various graphic design and layering tasks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorImageV2:

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
                "color": ("STRING", {"default": "#000000"},),
            },
            "optional": {
                "size_as": (any, {}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'color_image_v2'
    CATEGORY = '😺dzNodes/LayerUtility'

    def color_image_v2(self, size, custom_width, custom_height, color, size_as=None ):

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

        ret_image = Image.new('RGB', (width, height), color=color)
        return (pil2tensor(ret_image), )

```
