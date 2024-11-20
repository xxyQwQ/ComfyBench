---
tags:
- Text
- TextReplacement
---

# String Replace
## Documentation
- Class name: `DF_String_Replace`
- Category: `Derfuu_Nodes/Functions/String Operations`
- Output node: `False`

The `DF_String_Replace` node is designed to modify strings by replacing specified patterns with a new string. It supports both strict replacements and regular expression-based substitutions, allowing for flexible text manipulation within a given text input.
## Input types
### Required
- **`Text`**
    - The original text in which replacements will be made. It serves as the base for modifications.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Pattern`**
    - The pattern to be replaced in the text. This can be a literal string (in strict mode) or a regular expression pattern.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Replace_With`**
    - The string that will replace occurrences of the pattern in the text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Mode`**
    - Determines whether the replacement will be done using strict matching or regular expressions, offering flexibility in how patterns are identified and replaced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`TEXT`**
    - Comfy dtype: `STRING`
    - The modified text after the specified patterns have been replaced.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringReplace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": Field.string(),
                "Pattern": Field.string(),
                "Replace_With": Field.string(),
                "Mode": Field.combo(["Strict", "RegEx"]),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("TEXT",)
    FUNCTION = "replace"
    CATEGORY = TREE_STRINGS

    def replace(self, Text: str, Pattern: str, Replace_With: str, Mode: str):
        out = Text
        Pattern = Pattern.encode().decode("unicode_escape")
        match Mode:
            case "Strict":
                out = Text.replace(Pattern, Replace_With)
            case "RegEx":
                out = re.sub(Pattern, Replace_With, out, flags=re.MULTILINE)
        return (out,)

```
