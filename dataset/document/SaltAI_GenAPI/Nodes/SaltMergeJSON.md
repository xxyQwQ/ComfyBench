# Merge JSON
## Documentation
- Class name: `SaltMergeJSON`
- Category: `SALT/JSON`
- Output node: `False`

The SaltMergeJSON node is designed to merge two JSON strings into a single JSON string, effectively combining their contents. This node is particularly useful in scenarios where integrating data from multiple JSON sources is required.
## Input types
### Required
- **`json_input1`**
    - The first JSON string to be merged. It plays a crucial role in the merging process as it forms the base onto which the second JSON string's contents are added.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`json_input2`**
    - The second JSON string to be merged with the first. Its contents are added to those of the first JSON string, contributing to the creation of the merged JSON output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The resulting JSON string after merging the contents of the two input JSON strings. It represents the combined data structure.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltMergeJSON:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input1": ("STRING", {"forceInput": True}),
                "json_input2": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "merge"
    CATEGORY = f"{MENU_NAME}/JSON"

    def merge(self, json_input1, json_input2):
        try:
            json_data1 = json.loads(json_input1)
            json_data2 = json.loads(json_input2)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        merged_json = {**json_data1, **json_data2}
        return (json.dumps(merged_json),)

```
