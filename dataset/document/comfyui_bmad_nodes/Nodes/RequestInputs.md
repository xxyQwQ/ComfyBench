---
tags:
- Text
---

# RequestInputs
## Documentation
- Class name: `RequestInputs`
- Category: `Bmad/api`
- Output node: `False`

The RequestInputs node is designed to process a string of values, typically received as a JSON string, and convert them into a tuple of strings. This node is essential for handling and parsing input data received from web requests or other input sources, making it a key component in preparing data for further processing or analysis within a system.
## Input types
### Required
- **`values`**
    - The 'values' parameter accepts a JSON string representing the input values. It plays a crucial role in the node's operation by determining the data that will be parsed and converted into a tuple of strings for further processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - Each element in the output tuple represents a string value extracted and parsed from the input JSON string, facilitating the handling of multiple input values simultaneously.
    - Python dtype: `Tuple[str, ...]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RequestInputs:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "values": ("STRING", {"default": ""}),
        },
        }

    RETURN_TYPES = tuple(["STRING" for x in range(32)])
    FUNCTION = "start"
    CATEGORY = api_category_path

    def start(self, values):
        values = tuple(json.loads(values).values())
        return values

```
