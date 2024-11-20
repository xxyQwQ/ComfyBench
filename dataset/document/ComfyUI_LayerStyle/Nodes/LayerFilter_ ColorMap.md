# LayerFilter: ColorMap
## Documentation
- Class name: `LayerFilter: ColorMap`
- Category: `😺dzNodes/LayerFilter`
- Output node: `False`

The ColorMap node applies a specified color map to an image, optionally adjusting its opacity, to transform the visual appearance of the image. This node supports a variety of color maps, allowing for a wide range of aesthetic effects.
## Input types
### Required
- **`image`**
    - The input image to which the color map will be applied. This is the primary data the node operates on, determining the visual output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`color_map`**
    - Specifies the color map to apply to the input image. The choice of color map affects the aesthetic and thematic presentation of the image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`opacity`**
    - Determines the opacity level of the color map applied to the image, allowing for fine-tuning of the visual effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after the color map and opacity adjustments have been applied, showcasing the transformed visual appearance.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorMap:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "color_map": (colormap_list,),
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_map'
    CATEGORY = '😺dzNodes/LayerFilter'

    def color_map(self, image, color_map, opacity
                  ):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i, 0)
            _canvas = tensor2pil(i)
            _image = image_to_colormap(_canvas, colormap_list.index(color_map))
            ret_image = chop_image(_canvas, _image, 'normal', opacity)

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
