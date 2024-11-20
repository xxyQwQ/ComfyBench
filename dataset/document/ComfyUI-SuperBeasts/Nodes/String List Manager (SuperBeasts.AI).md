# String List Manager (SuperBeasts.AI)
## Documentation
- Class name: `String List Manager (SuperBeasts.AI)`
- Category: `SuperBeastsAI/Utils`
- Output node: `False`

The String List Manager node is designed to reorder a list of strings based on a specified new order. It allows for dynamic reorganization of string elements, facilitating customized sequencing or prioritization of data within a list.
## Input types
### Required
- **`frames_per_image`**
    - Specifies the number of frames to be associated with each image, affecting the output string format by determining the numerical prefix for each string.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`new_order`**
    - Defines a custom order for the strings to be rearranged. If provided, strings are reordered according to the indices specified in this parameter.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Outputs a reordered list of strings, formatted with frame numbers as prefixes, based on the input order and frames per image.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringListManager:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "frames_per_image": ("INT", {"default": 1, "min": 1, "step": 1})
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "reorder_strings"
    CATEGORY = "SuperBeastsAI/Utils"

    def reorder_strings(self, frames_per_image, new_order, **kwargs):
        strings = [kwargs["string1"]]  # Start with the required string1 input

        i = 2
        while f"string{i}" in kwargs:
            strings.append(kwargs[f"string{i}"])
            i += 1

        if new_order:
            order_indices = [int(idx) - 1 for idx in new_order.split(',') if idx.strip()]
            strings = [strings[idx] for idx in order_indices if idx < len(strings)]

        result = []
        for i, string in enumerate(strings):
            result.append('"{frames}": "{string}"'.format(frames=frames_per_image * i, string=string))

        return (",\n".join(result),)

```
