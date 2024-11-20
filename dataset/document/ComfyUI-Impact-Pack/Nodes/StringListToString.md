# String List to String
## Documentation
- Class name: `StringListToString`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to concatenate a list of strings into a single string, using a specified delimiter. It allows for flexible string manipulation, particularly useful for aggregating multiple pieces of text with custom spacing or formatting.
## Input types
### Required
- **`join_with`**
    - Specifies the delimiter to use for joining the list of strings. The default delimiter is a newline character, allowing for easy creation of multi-line strings from a list.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`string_list`**
    - The list of strings to be concatenated. This input is marked as required, emphasizing its critical role in the node's operation.
    - Comfy dtype: `STRING`
    - Python dtype: `List[str]`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The concatenated string, formed by joining the input list of strings with the specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringListToString:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "join_with": ("STRING", {"default": "\\n"}),
                "string_list": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    def doit(self, join_with, string_list):
        # convert \\n to newline character
        if join_with[0] == "\\n":
            join_with[0] = "\n"

        joined_text = join_with[0].join(string_list)

        return (joined_text,)

```
