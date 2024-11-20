# JSON Values to List
## Documentation
- Class name: `SaltJSONValuesToList`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to extract the values from a JSON object and provide them in two formats: as a list and as a string with each value separated by a newline. It ensures that the input JSON is properly formatted and is a dictionary before proceeding with the extraction.
## Input types
### Required
- **`json_input`**
    - The JSON string input from which values are to be extracted. It must be a well-formed JSON dictionary. This input is crucial for the node's operation as it determines the source of the values to be processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`values`**
    - Comfy dtype: `LIST`
    - A list of values extracted from the input JSON dictionary.
    - Python dtype: `List[Any]`
- **`values_list`**
    - Comfy dtype: `STRING`
    - A string representation of the values extracted from the input JSON, separated by newlines.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONValuesToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("values", "values_list")
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "values_list"
    CATEGORY = f"{MENU_NAME}/JSON"

    def values_list(self, json_input):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        if not isinstance(json_data, dict):
            raise Exception("Input JSON must be a dictionary")

        values_list = list(json_data.values())
        values_string = "\n".join(map(str, values_list))

        return (values_list, values_string)

```
