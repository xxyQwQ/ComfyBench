# LayerUtility: XY to Percent
## Documentation
- Class name: `LayerUtility: XY to Percent`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node converts the X and Y coordinates of a layer image relative to a background image into percentage values, facilitating the positioning of the layer image on the background image in a scalable manner.
## Input types
### Required
- **`background_image`**
    - The background image against which the layer image's position is calculated. It serves as the reference frame for determining the percentage-based position.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`layer_image`**
    - The layer image whose position is to be calculated relative to the background image. This image's dimensions are used in the calculation of its center's position in percentage terms.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`x`**
    - The X coordinate of the layer image's position, which is used along with the background image's dimensions to calculate the horizontal position in percentage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y`**
    - The Y coordinate of the layer image's position, which is used along with the background image's dimensions to calculate the vertical position in percentage.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`x_percent`**
    - Comfy dtype: `FLOAT`
    - The calculated horizontal position of the layer image's center as a percentage of the background image's width.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class XYtoPercent:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),  #
                "layer_image": ("IMAGE",),  #
                "x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT",)
    RETURN_NAMES = ("x_percent", "x_percent",)
    FUNCTION = 'xy_to_percent'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def xy_to_percent(self, background_image, layer_image, x, y,):

        _canvas = tensor2pil(background_image).convert('RGB')
        _layer = tensor2pil(layer_image).convert('RGB')
        x_percent = (x + _layer.width / 2) / _canvas.width * 100.0
        y_percent = (y + _layer.height / 2) / _canvas.height * 100.0

        return (x_percent, y_percent,)

```
