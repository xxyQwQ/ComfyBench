# Split String by Newline
## Documentation
- Class name: `SaltSplitByNewline`
- Category: `SALT/String`
- Output node: `False`

The node splits a given text into a list of strings based on newline characters, optionally stripping whitespace and ignoring lines that start with common comment symbols (#, //).
## Input types
### Required
- **`text`**
    - The text to be split into lines. This input allows for the processing of multiline text, enabling the node to operate on extensive or compact text data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`strip_text`**
    - A boolean flag that determines whether whitespace should be stripped from the beginning and end of each line, as well as ignoring lines that start with comment symbols.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`list`**
    - Comfy dtype: `LIST`
    - A list of strings derived from the input text, split based on newline characters and optionally processed according to the strip_text parameter.
    - Python dtype: `List[str]`
- **`strings`**
    - Comfy dtype: `STRING`
    - A duplicate of the list output, provided for compatibility or further processing needs.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSplitByNewline:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "strip_text": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("list", "strings")
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "split"
    CATEGORY = f"{MENU_NAME}/String"

    def split(self, text, strip_text):
        if strip_text:
            text = text.strip()
            lines = [line.strip() for line in text.splitlines() if line and (not line.strip().startswith("#") or not line.strip().startswith("//"))]
        else:
            lines = [line for line in text.splitlines() if line and (not line.strip().startswith("#") or not line.strip().startswith("//"))]
        return (lines, lines)

```
