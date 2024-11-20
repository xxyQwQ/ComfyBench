# Select Nth Items from JSON
## Documentation
- Class name: `SaltJSONSelectNthItems`
- Category: `SALT/JSON`
- Output node: `False`

This node is designed to select a specified number of items starting from a given index within a JSON object. It aims to facilitate the extraction of specific portions of JSON data based on positional selection, making it easier to work with subsets of JSON structures.
## Input types
### Required
- **`json_input`**
    - The JSON string from which items will be selected. It is crucial as it provides the data that the node operates on, determining the scope of selection.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_index`**
    - The starting index from which items will be selected. It defines the beginning of the selection range within the JSON object.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`count`**
    - The number of items to be selected starting from the start_index. It specifies the size of the subset to be extracted from the JSON object.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`selection`**
    - Comfy dtype: `STRING`
    - The JSON string representing the selected items. It is the result of extracting a subset of items from the input JSON based on the specified start index and count.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONSelectNthItems:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "start_index": ("INT", {"default": 0, "min": 0}),
                "count": ("INT", {"default": 1, "min": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selection",)

    FUNCTION = "select_nth"
    CATEGORY = f"{MENU_NAME}/JSON"

    def select_nth(self, json_input, start_index, count):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        if not isinstance(json_data, dict):
            raise Exception("Input JSON must be a dictionary")

        items = list(json_data.items())
        selected_items = items[start_index:start_index + count]
        selected_dict = dict(selected_items)

        return (json.dumps(selected_dict),)

```
