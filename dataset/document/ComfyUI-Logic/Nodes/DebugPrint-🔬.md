# DebugPrint
## Documentation
- Class name: `DebugPrint-🔬`
- Category: `Logic`
- Output node: `True`

The DebugPrint node is designed for logging purposes, specifically to print any given input to the console. This functionality aids in debugging by allowing the visualization of data flow through the node.
## Input types
### Required
- **`ANY`**
    - Represents any input that the node will log to the console. Its flexibility allows for a wide range of data types to be printed, aiding in debugging and data flow analysis.
    - Comfy dtype: `{}`
    - Python dtype: `Any`
## Output types
The node doesn't have output types
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DebugPrint:
    """
    This node prints the input to the console.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Takes in any input.

        """
        return {"required": {"ANY": (AlwaysEqualProxy({}),)}}

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "log_input"

    CATEGORY = "Logic"

    def log_input(self, ANY):
        print(ANY)
        return {}

```
