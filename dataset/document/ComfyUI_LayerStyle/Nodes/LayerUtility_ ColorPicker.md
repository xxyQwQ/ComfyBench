# LayerUtility: ColorPicker
## Documentation
- Class name: `LayerUtility: ColorPicker`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ColorPicker node provides functionality for selecting colors and converting them between different formats, such as hexadecimal and decimal color codes. It allows users to input a color in a specified format and convert it to another, facilitating easy integration and manipulation of color data within a design or visualization workflow.
## Input types
### Required
- **`color`**
    - Specifies the color to be converted or selected. It serves as the primary input for the node, determining the base color for conversion or selection.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`mode`**
    - Determines the output format of the color conversion, allowing users to choose between hexadecimal ('HEX') and decimal ('DEC') representations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
### Optional
## Output types
- **`value`**
    - Comfy dtype: `STRING`
    - The converted or selected color value, outputted in the format specified by the 'mode' input.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorPicker:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        mode_list = ['HEX', 'DEC']
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
                "mode": (mode_list,),  # 输出模式
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("value",)
    FUNCTION = 'picker'
    CATEGORY = '😺dzNodes/LayerUtility'

    def picker(self, color, mode):
        ret = color
        if mode == 'DEC':
            ret = Hex_to_RGB(ret)
        return (ret,)

```
