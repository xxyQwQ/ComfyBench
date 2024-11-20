# 📃 IG Float List
## Documentation
- Class name: `IG Float List`
- Category: `🐓 IG Nodes/Primitives`
- Output node: `False`

The IG Float List node is designed for parsing and organizing lists of floating-point numbers from string inputs. It enables precise control over the representation of these numbers, including the selection of specific elements based on index and the formatting of the entire list to a specified number of decimal places. This functionality is essential for tasks that require manipulation and detailed examination of numerical data within a user interface.
## Input types
### Required
- **`index`**
    - Specifies the position in the float list from which a single float value will be returned. It plays a crucial role in selecting specific elements for further operations or display.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`decimal_places`**
    - Determines the number of decimal places to which the float values in the list will be formatted. This allows for precise control over the numerical representation, affecting both the output's appearance and its numerical accuracy.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`float_list`**
    - The string input containing the list of floats, separated by common delimiters. This input is parsed to create a list of float values, which are then processed according to the specified parameters.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`current value`**
    - Comfy dtype: `FLOAT`
    - The float value at the specified index within the list. It provides direct access to individual elements, facilitating detailed numerical analysis.
    - Python dtype: `float`
- **`list`**
    - Comfy dtype: `FLOAT`
    - The list of parsed float values. This output offers a structured representation of the input data, enabling further numerical operations or analysis.
    - Python dtype: `list[float]`
- **`list text`**
    - Comfy dtype: `STRING_LIST`
    - A list of strings representing the float values formatted to the specified number of decimal places. This output is useful for displaying numerical data in a user-friendly format.
    - Python dtype: `list[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IG_FloatList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index": ( "INT", {"default": 1, "min": 1, "max": 100 } ),
                "decimal_places": ("INT", {"default": 3, "min": 1}),
                "float_list": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("FLOAT","FLOAT","STRING_LIST")
    RETURN_NAMES = ("current value", "list", "list text")
    FUNCTION = "ParseAndReturnFloat"
    CATEGORY = TREE_PRIMITIVES

    def __init__(self):
        self.static_text = "" 
        self.static_out_arr = []

    def ParseAndReturnFloat( self, index: int, decimal_places: int, float_list: str ) -> tuple[float, list[float], list[str]]:
        if float_list != self.static_text:
            split_str = re.split( ",|;|\s|:", float_list )
            out_arr = []
            for val in split_str:
                # let the exception happen if invalid
                out_arr.append(float(val))
            self.static_text = float_list
            self.static_out_arr = deepcopy( out_arr )
        ret_val = 0.0
        if ( index > len(self.static_out_arr) ):
            ret_val = self.static_out_arr[-1]
        else:
            ret_val = self.static_out_arr[ index - 1 ]
        return (ret_val,self.static_out_arr, ["{:.{}f}".format(val, decimal_places) for val in self.static_out_arr])

```
