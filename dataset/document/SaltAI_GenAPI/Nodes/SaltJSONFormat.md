# Format JSON
## Documentation
- Class name: `SaltJSONFormat`
- Category: `SALT/JSON`
- Output node: `False`

The SaltJSONFormat node is designed to reformat JSON data according to specified formatting options such as indentation, width, depth, compactness, and dictionary sorting. This node aims to enhance readability or compactness of JSON data for better handling and visualization.
## Input types
### Required
- **`json_input`**
    - The raw JSON string that needs to be reformatted. It is essential for defining the structure and content of the data to be processed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`indent`**
    - Specifies the number of spaces to use for indentation in the formatted JSON, allowing customization of the visual structure.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Sets the maximum width of each line in the formatted JSON, controlling line breaks for readability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`depth`**
    - Determines the maximum depth to which the JSON will be formatted, affecting the visibility of nested structures.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`compact`**
    - A boolean flag that, when true, minimizes the use of whitespace in the formatted JSON to produce a more compact output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`sort_dicts`**
    - Controls whether dictionaries within the JSON are sorted by key, aiding in consistent and organized output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`json_output`**
    - Comfy dtype: `STRING`
    - The reformatted JSON string, which has been adjusted according to the input parameters for improved structure and readability.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltJSONFormat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("STRING", {"forceInput": True}),
                "indent": ("INT", {"default": 4, "min": 0, "max": 8}),
                "width": ("INT", {"default": 80, "min": 16, "max": 4096}),
                "depth": ("INT", {"default": 1024, "min": 1, "max": 99999}),
                "compact": ("BOOLEAN", {"default": False}),
                "sort_dicts": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_output",)

    FUNCTION = "format"
    CATEGORY = f"{MENU_NAME}/JSON"

    def format(self, json_input, indent, width, depth, compact, sort_dicts):
        try:
            json_data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise Exception(f"Error in JSON formatting: {e}")

        json_formatted = pformat(json_data, indent=indent, width=width, depth=depth, compact=compact, sort_dicts=sort_dicts)

        return (json_formatted, )

```
