# Join String Multi
## Documentation
- Class name: `JoinStringMulti`
- Category: `KJNodes`
- Output node: `False`

The JoinStringMulti node dynamically combines multiple input strings into a single string or a list of strings, based on user-defined parameters such as the number of inputs and a delimiter. It allows for flexible string manipulation within a flow, accommodating a variable number of string inputs.
## Input types
### Required
- **`inputcount`**
    - Specifies the number of input strings to be combined. This allows the node to dynamically adjust to the desired number of inputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`string_i`**
    - Represents an input string to be combined. This serves as a generic placeholder for any string input beyond the first, facilitating the concatenation process with variable input counts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`delimiter`**
    - The delimiter used to separate the input strings in the final combined string.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`return_list`**
    - Determines whether the output should be a single concatenated string or a list of the individual input strings.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The result of combining the input strings, either as a single concatenated string or a list of strings, depending on the return_list parameter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class JoinStringMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "string_1": ("STRING", {"default": '', "forceInput": True}),
                "string_2": ("STRING", {"default": '', "forceInput": True}),
                "delimiter": ("STRING", {"default": ' ', "multiline": False}),
                "return_list": ("BOOLEAN", {"default": False}),
            },
    }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Creates single string, or a list of strings, from  
multiple input strings.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, delimiter, **kwargs):
        string = kwargs["string_1"]
        return_list = kwargs["return_list"]
        strings = [string] # Initialize a list with the first string
        for c in range(1, inputcount):
            new_string = kwargs[f"string_{c + 1}"]
            if return_list:
                strings.append(new_string) # Add new string to the list
            else:
                string = string + delimiter + new_string
        if return_list:
            return (strings,) # Return the list of strings
        else:
            return (string,) # Return the combined string

```
