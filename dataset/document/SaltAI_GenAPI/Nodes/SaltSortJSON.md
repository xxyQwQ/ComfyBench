# Sort JSON
## Documentation
- Class name: `SaltSortJSON`
- Category: `SALT/JSON`
- Output node: `False`

This node sorts a list of dictionaries contained within a JSON string based on a specified key and order. It is designed to organize JSON data in a more readable or structured format, facilitating easier data manipulation and analysis.
## Input types
### Required
- **`json_input`**
    - The JSON string input that contains a list of dictionaries to be sorted. It is essential for specifying the data to be organized.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`sort_key`**
    - The key within the dictionaries to sort the list by. This key determines the order of the sorted data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`reverse`**
    - A boolean flag that specifies the sorting order. When true, the list is sorted in descending order; otherwise, it's sorted in ascending order.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The sorted JSON string output, providing a structured representation of the input data after sorting.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSortJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "sort_key": ("STRING", {"multiline": False}),
                "reverse": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "sort"
    CATEGORY = f"{MENU_NAME}/JSON"

    def sort(self, json_input, sort_key, reverse):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        if isinstance(json_data, list):
            sorted_data = sorted(json_data, key=lambda x: x.get(sort_key), reverse=reverse)
            return (json.dumps(sorted_data),)
        else:
            raise Exception("Input JSON must be a list of dictionaries")

```
