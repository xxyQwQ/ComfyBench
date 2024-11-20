# textCycleLine
## Documentation
- Class name: `ttN textCycleLine`
- Category: `🌏 tinyterra/text`
- Output node: `False`

The ttN textCycleLine node is designed for cycling through lines of text based on a given index and control method. It allows for dynamic text manipulation by selecting specific lines from a larger text body, supporting operations like increment, decrement, random selection, or fixed position access.
## Input types
### Required
- **`text`**
    - The 'text' parameter is the input text from which lines will be cycled through. It plays a crucial role in determining the content available for cycling and directly impacts the output based on the selected line.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`index`**
    - The 'index' parameter specifies the starting point or specific line to access within the input text. Its value influences which line is selected during the cycling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`index_control`**
    - The 'index_control' parameter determines the method of cycling through lines, such as incrementing, decrementing, randomizing, or fixing the index. This affects how the next line is selected from the input text.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_textCycleLine:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "index_control": (['increment', 'decrement', 'randomize','fixed'],),
                    },
                "hidden": {"ttNnodeVersion": ttN_textCycleLine.version},
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "cycle"

    CATEGORY = "🌏 tinyterra/text"

    def cycle(self, text, index, index_control='randomized'):
        lines = text.split('\n')

        if index >= len(lines):
            index = len(lines) - 1
        return (lines[index],)

```
