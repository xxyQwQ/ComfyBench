# Flatten JSON
## Documentation
- Class name: `SaltFlattenJSON`
- Category: `SALT/JSON`
- Output node: `False`

The SaltFlattenJSON node is designed to transform nested JSON structures into a flat dictionary format, where nested keys are combined into single keys separated by underscores. This simplification process aids in the easier handling and processing of complex JSON data by reducing its hierarchical depth.
## Input types
### Required
- **`json_input`**
    - The JSON input that needs to be flattened. This parameter is crucial as it directly influences the node's ability to simplify complex JSON structures into a more manageable form.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The output of the node is a JSON string that represents the original JSON input in a flattened format. This makes the data easier to work with in environments that may not support deeply nested structures.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltFlattenJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "flatten"
    CATEGORY = f"{MENU_NAME}/JSON"

    def flatten(self, json_input):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)

        flattened_json = flatten_dict(json_data)
        return (json.dumps(flattened_json),)

```
