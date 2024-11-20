# debugInput
## Documentation
- Class name: `ttN debugInput`
- Category: `🌏 tinyterra/util`
- Output node: `True`

The node is designed for debugging purposes, allowing users to input various parameters and observe their effects in a controlled environment. It primarily focuses on capturing and displaying input values to facilitate troubleshooting and analysis of data flow within the system.
## Input types
### Required
- **`print_to_console`**
    - Determines whether the debug information should be printed to the console, enabling or disabling console output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`console_title`**
    - Specifies the title to be displayed on the console during debugging, serving as an identifier for the debug session.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`console_color`**
    - Specifies the color in which the console text is displayed, allowing for visual differentiation of debug outputs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`debug`**
    - An optional parameter for additional debugging information, enhancing the granularity of the debug output.
    - Comfy dtype: ``
    - Python dtype: `Optional[str]`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_debugInput:
    version = '1.0.0'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ("BOOLEAN",),
                    "console_title": ("STRING", {"default": "ttN debug:"}),
                    "console_color": (["Black", "Red", "Green", "Yellow", "Blue", "Violet", "Cyan", "White", "Grey", "LightRed", "LightGreen", "LightYellow", "LightBlue", "LightViolet", "LightCyan", "LightWhite"], {"default": "Red"}),
                    },
                "optional": {
                    "debug": ("", {"default": None}),
                    }
        }

    RETURN_TYPES = tuple()
    RETURN_NAMES = tuple()
    FUNCTION = "debug"
    CATEGORY = "🌏 tinyterra/util"
    OUTPUT_NODE = True

    def debug(_, print_to_console, console_title, console_color, debug=None):

        text = str(debug)
        if print_to_console:
            print(f"{getattr(CC, console_color.upper())}{console_title}\n{text}{CC.CLEAN}")

        return {"ui": {"text": text}, "return": tuple()}

```
