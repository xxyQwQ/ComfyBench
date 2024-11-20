---
tags:
- VisualEffects
---

# LayerFilter: WaterColor
## Documentation
- Class name: `LayerFilter: WaterColor`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The WaterColor node applies a watercolor effect to images, transforming their appearance with customizable line density and opacity levels to simulate the look of watercolor paintings.
## Input types
### Required
- **`image`**
    - The input image to which the watercolor effect will be applied. This parameter is crucial for defining the base image that will undergo the transformation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`line_density`**
    - Controls the density of the lines in the watercolor effect, affecting the overall texture and detail of the resulting image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`opacity`**
    - Determines the opacity of the watercolor effect, allowing for fine-tuning of the visual impact on the original image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with the watercolor effect applied, showcasing the transformation from the original input.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class WaterColor:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "line_density": ("INT", {"default": 50, "min": 1, "max": 100, "step": 1}),  # 透明度
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'water_color'
    CATEGORY = '😺dzNodes/LayerFilter'

    def water_color(self, image, line_density, opacity
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i).convert('RGB')
            _image = image_watercolor(_canvas, level=101-line_density)
            ret_image = chop_image(_canvas, _image, 'normal', opacity)

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
