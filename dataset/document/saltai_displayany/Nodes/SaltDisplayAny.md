---
tags:
- DataTypeAgnostic
- Debugging
---

# SaltDisplayAny
## Documentation
- Class name: `SaltDisplayAny`
- Category: `SALT/Utility`
- Output node: `True`

The SaltDisplayAny node is designed to universally display any type of input data, converting it into a human-readable format or indicating if the data type cannot be displayed. It supports a wide range of data types, including basic types like strings and numbers, complex data structures like dictionaries and lists, and even PyTorch tensors, making it a versatile utility for debugging and data visualization.
## Input types
### Required
- **`input_value`**
    - Represents the data to be displayed. It can be of any type, making this node highly versatile in handling various kinds of input for visualization or debugging purposes.
    - Comfy dtype: `*`
    - Python dtype: `AnyType`
## Output types
- **`output`**
    - Comfy dtype: `*`
    - The output is a human-readable representation of the input data or a message indicating that the data type cannot be displayed.
    - Python dtype: `str`
- **`ui`**
    - A UI component that displays the text representation of the input data, facilitating easy visualization in a user interface.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SaltDisplayAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_value": (WILDCARD, {}),
            },
        }

    RETURN_TYPES = (WILDCARD, )
    RETURN_NAMES = ("output", )
    WEB_DIRECTORY = "./web"

    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "SALT/Utility"

    def main(self, input_value=None):
        if input_value is None:
            return {"ui": {"text":""}, "result": ("",)}

        if isinstance(input_value, (str, int, float, bool)):
            value = str(input_value)
        elif isinstance(input_value, (dict, list)):
            try:
                value = json.dumps(input_value)
            except Exception:
                value = "Data could not be serialized."
        elif isinstance(input_value, torch.Tensor):
            value = str(input_value.shape)
        else:
            value = f"Data type {type(input_value).__name__} cannot be displayed"

        logging.info(value)
        return {"ui": {"text": value}, "result": (input_value,)}

```
