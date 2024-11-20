# Int
## Documentation
- Class name: `Int-🔬`
- Category: `Logic`
- Output node: `False`

The Int node is designed to process and output integer values. It serves as a fundamental logic unit within a system, enabling the manipulation and handling of integer data types for various computational and logical operations.
## Input types
### Required
- **`value`**
    - Specifies the integer value to be processed by the node. This parameter is central to the node's operation, as it determines the exact integer data that will be manipulated or evaluated.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`INT`**
    - Comfy dtype: `INT`
    - Outputs the integer value that was input, after any processing. This demonstrates the node's ability to handle and return integer data, maintaining its integrity throughout the operation.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Int:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"value": ("INT", {"default": 0})},
        }

    RETURN_TYPES = ("INT",)

    RETURN_NAMES = ("INT",)

    FUNCTION = "execute"

    CATEGORY = "Logic"

    def execute(self, value):
        return (value,)

```
