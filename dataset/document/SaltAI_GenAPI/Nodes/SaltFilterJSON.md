# Filter JSON
## Documentation
- Class name: `SaltFilterJSON`
- Category: `SALT/JSON`
- Output node: `False`

The SaltFilterJSON node is designed to filter elements from a JSON array based on a specified condition. It evaluates each element against a given key-value pair and returns a new JSON array containing only those elements that match the condition.
## Input types
### Required
- **`json_input`**
    - The JSON string input that represents an array of objects. It is the primary data source for the filtering operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`condition_key`**
    - The key to be checked within each object of the JSON array. It determines the field against which the condition is applied.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`condition_value`**
    - The value that the specified key's value must match for an object to be included in the filtered output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The output is a JSON string representing an array of objects that meet the specified condition.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltFilterJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "condition_key": ("STRING", {"multiline": False}),
                "condition_value": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "filter"
    CATEGORY = f"{MENU_NAME}/JSON"

    def filter(self, json_input, condition_key, condition_value):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        filtered_data = [item for item in json_data if item.get(condition_key) == condition_value]
        return (json.dumps(filtered_data),)

```
