# ⌨️ IG ZFill
## Documentation
- Class name: `IG ZFill`
- Category: `🐓 IG Nodes/Primitives`
- Output node: `False`

The IG ZFill node is designed to pad a given integer value with zeros up to a specified length. It is useful for formatting numbers to ensure they meet a certain string length requirement, enhancing readability or meeting specific data format standards.
## Input types
### Required
- **`value`**
    - The integer value to be padded with leading zeros. This parameter determines the number to be formatted.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fill`**
    - Specifies the total length of the resulting string after padding with zeros. It determines how many leading zeros are added to the input value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The formatted string with the input value padded with leading zeros to meet the specified total length.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IG_ZFill:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -sys.maxsize, "max": sys.maxsize, "step": 1}),
                "fill": ("INT", {"default": 6, "min": 0, "max": 8, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "main"
    CATEGORY = TREE_PRIMITIVES

    def main(self, value, fill):
        return (f"{value}".zfill(fill),)

```
