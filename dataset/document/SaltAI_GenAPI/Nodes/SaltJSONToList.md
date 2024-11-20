# JSON to List
## Documentation
- Class name: `SaltJSONToList`
- Category: `SALT/JSON`
- Output node: `False`

Converts a JSON string into a list of its innermost values, providing both a string representation and the actual list of these values. This node is designed to extract and present the deepest nested values found within a JSON structure, making them accessible for further processing or analysis.
## Input types
### Required
- **`json_input`**
    - The JSON string to be processed. This input is crucial as it provides the raw data from which innermost values are extracted, directly influencing the output list and its string representation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`values`**
    - Comfy dtype: `STRING`
    - A string representation of the list containing the innermost values extracted from the JSON input.
    - Python dtype: `str`
- **`values_list`**
    - Comfy dtype: `LIST`
    - The actual list of innermost values extracted from the JSON input.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("values", "values_list")
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "val_list"
    CATEGORY = f"{MENU_NAME}/JSON"

    def val_list(self, json_input):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        values_list = list(get_innermost_values(json_data))

        return (str(values_list), values_list)

```
