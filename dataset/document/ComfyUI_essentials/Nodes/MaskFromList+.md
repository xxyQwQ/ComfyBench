---
tags:
- Mask
- MaskList
---

# 🔧 Mask From List
## Documentation
- Class name: `MaskFromList+`
- Category: `essentials/mask`
- Output node: `False`

The MaskFromList+ node generates a mask based on a list of values, allowing for the creation of custom masks with specified dimensions and values. It supports both numerical and string inputs for mask values, providing flexibility in defining the mask's content.
## Input types
### Required
- **`width`**
    - Specifies the width of the generated mask. It determines the horizontal dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the generated mask. It determines the vertical dimension of the mask.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`values`**
    - A list of numerical values (integers or floats) or a single numerical value used to populate the mask. This input allows for the customization of the mask's intensity or transparency.
    - Comfy dtype: `INT,FLOAT`
    - Python dtype: `List[Union[int, float]] | Union[int, float]`
- **`str_values`**
    - A string representation of numerical values, separated by commas, used to populate the mask. This provides an alternative method for defining the mask's content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a mask created from the provided list of values, with the specified dimensions and value intensities.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskFromList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 32, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
            }, "optional": {
                "values": ("INT,FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, }),
                "str_values": ("STRING", { "default": "", "multiline": True, "placeholder": "0.0, 0.5, 1.0",}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, width, height, values=None, str_values=""):
        out = []

        if values is not None:
            if not isinstance(values, list):
                out = [values]
            else:
                out.extend(values)

        if str_values != "":
            str_values = [float(v) for v in str_values.split(",")]
            out.extend(str_values)

        if out == []:
            raise ValueError("No values provided")
        
        out = torch.tensor(out).float().clamp(0.0, 1.0)
        out = out.view(-1, 1, 1).expand(-1, height, width)
        
        values = None
        str_values = ""

        return (out, )

```
