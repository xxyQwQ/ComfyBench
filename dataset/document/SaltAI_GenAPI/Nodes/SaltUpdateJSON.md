# Update JSON
## Documentation
- Class name: `SaltUpdateJSON`
- Category: `SALT/JSON`
- Output node: `False`

The SaltUpdateJSON node is designed to update specific values within a JSON structure based on a given key. It recursively searches through the JSON, updating all instances of the key with a new value, thereby allowing for comprehensive modifications of nested JSON objects.
## Input types
### Required
- **`json_input`**
    - The JSON string to be updated. It serves as the base structure for modifications, where specific values associated with a given key are targeted for updates.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`key`**
    - The key whose associated values are to be updated across the JSON structure. It acts as the identifier for locating and modifying specific data points within the JSON.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`new_value`**
    - The new value to replace existing values associated with the specified key within the JSON structure. This allows for dynamic updates to the JSON's content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The updated JSON string, reflecting the changes made to values associated with the specified key. It represents the final state of the JSON after applying the updates.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltUpdateJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"multiline": False}),
                "new_value": ("STRING", {"multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "update"
    CATEGORY = f"{MENU_NAME}/JSON"

    def update(self, json_input, key, new_value):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        def recursive_update(data, key, new_value):
            if isinstance(data, dict):
                if key in data:
                    data[key] = new_value
                for k, v in data.items():
                    recursive_update(v, key, new_value)
            elif isinstance(data, list):
                for item in data:
                    recursive_update(item, key, new_value)

        recursive_update(json_data, key, new_value)
        return (json.dumps(json_data),)

```
