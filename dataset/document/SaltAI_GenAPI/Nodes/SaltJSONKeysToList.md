# JSON Keys to List
## Documentation
- Class name: `SaltJSONKeysToList`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to extract the keys from a JSON object and provide them as both a list and a concatenated string. It serves the purpose of enabling users to easily access and manipulate the structure of JSON data by focusing on its keys.
## Input types
### Required
- **`json_input`**
    - The JSON input as a string. It is required to be a valid JSON object (dictionary) for the operation to proceed. This input is crucial as it determines the keys that will be extracted and returned in the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`keys`**
    - Comfy dtype: `LIST`
    - A list of keys extracted from the input JSON object.
    - Python dtype: `List[str]`
- **`keys_list`**
    - Comfy dtype: `STRING`
    - A string representation of the keys extracted from the input JSON object, concatenated and separated by newlines.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONKeysToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("keys", "keys_list")
    OUTPUT_IS_LIST = (True, False)

    FUNCTION = "keys_list"
    CATEGORY = f"{MENU_NAME}/JSON"

    def keys_list(self, json_input):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        if not isinstance(json_data, dict):
            raise Exception("Input JSON must be a dictionary")

        keys_list = list(json_data.keys())
        keys_string = "\n".join(keys_list)

        return (keys_list, keys_string)

```
