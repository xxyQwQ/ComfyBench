# Select from JSON
## Documentation
- Class name: `SaltSelectFromJSON`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to select a specific item from a JSON structure based on a given JSON path and match index. It allows for precise extraction of data from complex JSON inputs, facilitating targeted data retrieval and manipulation.
## Input types
### Required
- **`json_input`**
    - The JSON string from which data will be selected. It serves as the primary data source for the selection process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`json_path`**
    - The JSONPath expression used to specify the item to be selected from the JSON input. It defines the path to the target data within the JSON structure.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`match_index`**
    - The index of the match to be selected when multiple items match the JSONPath expression. It determines which item is chosen if the path yields more than one result.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`selection`**
    - Comfy dtype: `STRING`
    - The selected item from the JSON input, returned as a JSON string. It represents the outcome of the selection process based on the provided path and index.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltSelectFromJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "json_path": ("STRING", {"multiline": False}),
                "match_index": ("INT", {"default": 0, "min": 0, "max": 99999})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selection",)

    FUNCTION = "select"
    CATEGORY = f"{MENU_NAME}/JSON"

    def select(self, json_input, json_path, match_index):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")
        jsonpath_expr = parse(json_path)
        matches = [match.value for match in jsonpath_expr.find(json_data)]
        if matches:
            if match_index >= len(matches):
                match_index = len(matches) - 1
            return (json.dumps(matches[match_index]),)
        else:
            return ("{}",) 

```
