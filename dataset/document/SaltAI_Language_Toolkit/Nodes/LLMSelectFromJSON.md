# ∞ Select from JSON
## Documentation
- Class name: `LLMSelectFromJSON`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to select specific data from a JSON input based on a given JSON path. It allows for the extraction of data by parsing the JSON structure and applying a JSONPath expression to identify and return the desired piece of data at a specified index.
## Input types
### Required
- **`json_input`**
    - The JSON string from which data is to be extracted. It serves as the primary data source for the operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`json_path`**
    - A JSONPath expression specifying the path to the desired data within the JSON input. It determines the specific piece of data to be extracted.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`match_index`**
    - The index of the match to return when multiple matches are found for the JSONPath expression. It controls which piece of data is selected when the path yields multiple results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`selection`**
    - Comfy dtype: `STRING`
    - The selected data from the JSON input as a JSON string. It represents the result of applying the JSONPath expression to the input.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSelectFromJSON:
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
