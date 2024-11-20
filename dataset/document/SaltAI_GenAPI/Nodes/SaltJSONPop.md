# Pop from JSON (Delete)
## Documentation
- Class name: `SaltJSONPop`
- Category: `SALT/JSON`
- Output node: `False`

The SaltJSONPop node is designed to selectively remove specified keys from a JSON structure. It can operate recursively, allowing for the deletion of keys not only at the root level but also deeply nested within the JSON object. This functionality is particularly useful for cleaning or modifying JSON data by removing unnecessary, sensitive, or redundant information.
## Input types
### Required
- **`json_input`**
    - The JSON string from which keys will be removed. This input is crucial as it defines the structure and content that will be manipulated.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`key`**
    - The name of the key(s) to be removed from the JSON input. This parameter determines which parts of the JSON structure will be targeted for deletion.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`max_occurrences`**
    - Defines the maximum number of occurrences of the specified key to remove. If set to -1, there is no limit, allowing for the removal of all instances of the key.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The modified JSON string after the specified keys have been removed. This output reflects the changes made to the original JSON structure.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONPop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "key": ("STRING", {"multiline": False}),
                "max_occurrences": ("INT", {"default": -1, "min": -1, "max": 99999})  # -1 means no limit
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "pop_key"
    CATEGORY = f"{MENU_NAME}/JSON"

    def pop_key(self, json_input, key, max_occurrences):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        remaining_occurrences = max_occurrences if max_occurrences >= 0 else float('inf')
        self.recursive_pop(json_data, key, remaining_occurrences)

        return (json.dumps(json_data),)


    def recursive_pop(self, obj, key, remaining_occurrences):
        if isinstance(obj, dict):
            if key in obj and (remaining_occurrences != 0):
                obj.pop(key)
                remaining_occurrences -= 1 if remaining_occurrences > 0 else 0
            for k, v in obj.items():
                remaining_occurrences = self.recursive_pop(v, key, remaining_occurrences)
        elif isinstance(obj, list):
            for item in obj:
                remaining_occurrences = self.recursive_pop(item, key, remaining_occurrences)
        return remaining_occurrences

```
