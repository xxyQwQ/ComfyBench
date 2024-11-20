# Compare
## Documentation
- Class name: `Compare-🔬`
- Category: `Logic`
- Output node: `False`

The Compare node is designed to evaluate the relationship between two inputs based on a specified comparison operation (e.g., equality, inequality, greater than, less than). It abstracts the logic of comparison into a simple interface, allowing users to dynamically assess the conditions between inputs and derive a boolean result indicative of the comparison's outcome.
## Input types
### Required
- **`a`**
    - The first input to be compared. It plays a crucial role in determining the outcome of the comparison operation.
    - Comfy dtype: `*`
    - Python dtype: `UNKNOWN`
- **`b`**
    - The second input to be compared against the first. Its value is essential in evaluating the comparison operation.
    - Comfy dtype: `*`
    - Python dtype: `UNKNOWN`
- **`comparison`**
    - Specifies the type of comparison to perform between the two inputs (e.g., '==', '!=', '<', '>', '<=', '>='). This parameter dictates the logic applied in the comparison.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`B`**
    - Comfy dtype: `BOOLEAN`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class Compare:
    """
    This nodes compares the two inputs and outputs the result of the comparison.
    """

    @classmethod
    def INPUT_TYPES(s):
        """
        Comparison node takes two inputs, a and b, and compares them.
        """
        s.compare_functions = list(COMPARE_FUNCTIONS.keys())
        return {
            "required": {
                "a": (AlwaysEqualProxy("*"), {"default": 0}),
                "b": (AlwaysEqualProxy("*"), {"default": 0}),
                "comparison": (s.compare_functions, {"default": "a == b"}),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)

    RETURN_NAMES = "BOOLEAN"

    FUNCTION = "compare"

    CATEGORY = "Logic"

    def compare(self, a, b, comparison):
        """
        Compare two inputs and return the result of the comparison.

        Args:
            a (UNKNOWN): The first input.
            b (UNKNOWN): The second input.
            comparison (STRING): The comparison to perform. Can be one of "==", "!=", "<", ">", "<=", ">=".

        Returns:
            : The result of the comparison.
        """
        return (COMPARE_FUNCTIONS[comparison](a, b),)

```
