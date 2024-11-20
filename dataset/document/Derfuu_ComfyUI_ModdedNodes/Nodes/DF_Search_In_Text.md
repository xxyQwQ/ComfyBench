# Search In Text
## Documentation
- Class name: `DF_Search_In_Text`
- Category: `Derfuu_Nodes/Functions/String Operations`
- Output node: `False`

The `DF_Search_In_Text` node is designed to search for a specified pattern within a given text. It supports both strict and regular expression (RegEx) modes for pattern matching, and can optionally consider case sensitivity. This node provides a way to determine if the pattern exists in the text and to count the number of occurrences, offering flexibility for text analysis and processing tasks.
## Input types
### Required
- **`Text`**
    - The text in which to search for the specified pattern. It serves as the primary input for the search operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`Pattern`**
    - The pattern to search for within the text. This can be a literal string in strict mode or a regular expression in RegEx mode.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`ConsiderRegister`**
    - A boolean indicating whether to consider case sensitivity in the search. When false, the search is case-insensitive.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`Mode`**
    - Specifies the mode of pattern matching: 'Strict' for literal string matching or 'RegEx' for regular expression matching.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`BOOLEAN`**
    - Comfy dtype: `BOOLEAN`
    - Indicates whether the specified pattern was found in the text.
    - Python dtype: `bool`
- **`OCCURRENCES`**
    - Comfy dtype: `INT`
    - The number of times the specified pattern was found in the text.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SearchInText:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Text": Field.string(),
                "Pattern": Field.string(),
                "ConsiderRegister": Field.boolean(default=False),
                "Mode": Field.combo(["Strict", "RegEx"]),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT",)
    RETURN_NAMES = ("BOOLEAN", "OCCURRENCES",)
    FUNCTION = "search_in_text"
    CATEGORY = TREE_STRINGS

    def search_in_text(self, Text: str, Pattern: str, ConsiderRegister: bool, Mode: str) -> tuple:
        out = None
        occs = 0
        Pattern = Pattern.encode().decode("unicode_escape")
        if not ConsiderRegister:
            Text = Text.lower()
            Pattern = Pattern.lower()
        match Mode:
            case "Strict":
                while Pattern in Text:
                    out = True
                    occs += 1
                    Text = Text.replace(Pattern, "", 1)
            case "RegEx":
                occs = len(re.findall(Pattern, Text))
                out = bool(occs)
        return (out, occs)

```
