---
tags:
- Concatenate
- Text
---

# Multi Text Merge
## Documentation
- Class name: `Multi Text Merge`
- Category: `Hakkun`
- Output node: `False`

The Multi Text Merge node is designed to concatenate multiple text inputs into a single string, using a specified delimiter to separate each input. It allows for the flexible combination of up to six text strings, making it useful for generating composite text outputs from various sources.
## Input types
### Required
- **`delimiter`**
    - Specifies the string used to separate each text input in the concatenated output. This parameter allows for customization of the output format, enabling the use of different separators such as commas, spaces, or custom strings.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`s1`**
    - The first optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`s2`**
    - The second optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`s3`**
    - The third optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`s4`**
    - The fourth optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`s5`**
    - The fifth optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`s6`**
    - The sixth optional text input to be concatenated. It supports multiline input and is forced to be included in the node's UI.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The concatenated result of the input texts, separated by the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MultiTextMerge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": (TEXT_TYPE, {"default": ', ', "multiline": False}),
            },
            "optional": {
                "s1": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s2": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s3": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s4": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s5": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
                "s6": (TEXT_TYPE, {"default": '', "multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = (TEXT_TYPE,)
    FUNCTION = "concatenate_strings"

    CATEGORY = "Hakkun"

    def concatenate_strings(self, s1='', s2='', s3='', s4='', s5='', s6='', delimiter="_"):
        delimiter = delimiter.replace("\\n", "\n")
        strings=[s1, s2, s3, s4, s5, s6]
        strings = [s for s in strings if isOk(s)]
        concatenated_string = delimiter.join(strings)
        return concatenated_string,

```
