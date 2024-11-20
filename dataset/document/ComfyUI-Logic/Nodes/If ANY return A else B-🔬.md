# If ANY return A else B
## Documentation
- Class name: `If ANY return A else B-🔬`
- Category: `Logic`
- Output node: `False`

This node conditionally executes one of two provided operations based on a boolean input. It serves as a logical branch, allowing for dynamic flow control within a node network by executing the first operation if the condition is true, and the second operation if false.
## Input types
### Required
- **`ANY`**
    - Represents a boolean condition that determines which of the two operations, IF_TRUE or IF_FALSE, will be executed. Its evaluation to True or False directs the flow of execution accordingly.
    - Comfy dtype: `*`
    - Python dtype: `bool`
- **`IF_TRUE`**
    - The operation to be executed if the ANY input evaluates to True. This allows for conditional execution of node operations based on the provided boolean condition.
    - Comfy dtype: `*`
    - Python dtype: `Any`
- **`IF_FALSE`**
    - The operation to be executed if the ANY input evaluates to False, providing an alternative path for execution in the node's logic.
    - Comfy dtype: `*`
    - Python dtype: `Any`
## Output types
- **`?`**
    - Comfy dtype: `*`
    - The output is either the result of IF_TRUE or IF_FALSE operation, depending on the evaluation of the ANY input.
    - Python dtype: `Any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IfExecute:
    """
    This node executes IF_TRUE if ANY is True, otherwise it executes IF_FALSE.

    ANY can be any input, IF_TRUE and IF_FALSE can be any output.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (AlwaysEqualProxy("*"),),
                "IF_TRUE": (AlwaysEqualProxy("*"),),
                "IF_FALSE": (AlwaysEqualProxy("*"),),
            },
        }

    RETURN_TYPES = (AlwaysEqualProxy("*"),)

    RETURN_NAMES = "?"

    FUNCTION = "return_based_on_bool"

    CATEGORY = "Logic"

    def return_based_on_bool(self, ANY, IF_TRUE, IF_FALSE):
        return (IF_TRUE if ANY else IF_FALSE,)

```
