# 📃 IG String List
## Documentation
- Class name: `IG String List`
- Category: `🐓 IG Nodes/Primitives`
- Output node: `False`

The IG String List node is designed to split a multiline string input into a list of strings based on newline characters and return a specific string from the list and the entire list itself. It allows for dynamic text manipulation and retrieval within a user-defined list, facilitating operations on collections of strings.
## Input types
### Required
- **`index`**
    - Specifies the position in the list of strings to retrieve the specific string. It determines which string to return based on its position in the list, enhancing the node's flexibility in accessing elements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`string_list`**
    - A multiline string input that the node splits into a list of strings based on newline characters. This input is essential for creating the string list that the node operates on, enabling dynamic text manipulation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`current value`**
    - Comfy dtype: `STRING`
    - Returns the string at the specified index from the list of strings, allowing for targeted retrieval of text.
    - Python dtype: `str`
- **`list`**
    - Comfy dtype: `STRING`
    - Returns the entire list of strings derived from the input multiline string, providing access to all split text elements.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IG_StringList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "string_list": ("STRING", {"multiline": True}), 
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("current value","list")
    FUNCTION = "SplitAndReturnStrings"
    CATEGORY = TREE_PRIMITIVES

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def SplitAndReturnStrings( self, index: int, string_list: str ) -> tuple[str, list[str]]:
        if string_list != self.static_text:
            # unlike the numeric list nodes, we only want to split on newlines
            # TODO: support manual delimiter specification?
            split_str = re.split( "\n", string_list )
            self.static_text = string_list
            self.static_out_arr = deepcopy( split_str )
        if ( index > len(self.static_out_arr) ):
            return ( self.static_out_arr[len(self.static_out_arr) - 1], self.static_out_arr)
        return (self.static_out_arr[ index - 1 ], self.static_out_arr)

```
