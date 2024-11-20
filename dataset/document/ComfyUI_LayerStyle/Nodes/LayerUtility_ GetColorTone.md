# LayerUtility: GetColorTone
## Documentation
- Class name: `LayerUtility: GetColorTone`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

This node is designed to analyze an image and determine its dominant or average color tone, returning the result in both HEX and HSV formats. It provides a straightforward way for users to extract color information from images, which can be useful in various applications such as theming, design, and visual analysis.
## Input types
### Required
- **`image`**
    - The image to analyze for color tone. This is the primary input on which the color analysis is performed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the method of color analysis: 'main_color' for dominant color or 'average' for average color of the image. This choice affects the outcome of the color tone analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
## Output types
- **`RGB color in HEX`**
    - Comfy dtype: `STRING`
    - The dominant or average color of the image represented as a HEX code.
    - Python dtype: `str`
- **`HSV color in list`**
    - Comfy dtype: `LIST`
    - The dominant or average color of the image represented in HSV format, provided as a list.
    - Python dtype: `List[int]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetColorTone:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['main_color', 'average']
        return {
            "required": {
                "image": ("IMAGE", ),  #
                "mode": (mode_list,),  # 主色/平均色
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("RGB color in HEX", "HSV color in list")
    FUNCTION = 'get_color_tone'
    CATEGORY = '😺dzNodes/LayerUtility'

    def get_color_tone(self, image, mode,):
        if image.shape[0] > 0:
            image = torch.unsqueeze(image[0], 0)
        _canvas = tensor2pil(image).convert('RGB')
        _canvas = gaussian_blur(_canvas, int((_canvas.width + _canvas.height) / 200))
        if mode == 'main_color':
            ret_color = get_image_color_tone(_canvas)
        else:
            ret_color = get_image_color_average(_canvas)
        hsv_color = RGB_to_HSV(Hex_to_RGB(ret_color))

        return (ret_color, hsv_color)

```
