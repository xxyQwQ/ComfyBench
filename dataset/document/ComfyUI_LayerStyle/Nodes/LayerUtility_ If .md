# LayerUtility: If
## Documentation
- Class name: `LayerUtility: If `
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

This node provides conditional logic execution within a workflow, allowing for branching paths based on specified conditions. It evaluates a given condition and directs the flow accordingly, enabling dynamic decision-making in processing pipelines.
## Input types
### Required
- **`if_condition`**
    - Determines the branch of execution based on its truth value. A true condition leads to the execution of the 'when_TRUE' path, while a false condition leads to the 'when_FALSE' path.
    - Comfy dtype: `*`
    - Python dtype: `bool`
- **`when_TRUE`**
    - The path or operation to execute if the 'if_condition' evaluates to true. This allows for dynamic control over the workflow based on conditions.
    - Comfy dtype: `*`
    - Python dtype: `any`
- **`when_FALSE`**
    - The path or operation to execute if the 'if_condition' evaluates to false. This provides an alternative execution path, ensuring flexibility in handling various scenarios.
    - Comfy dtype: `*`
    - Python dtype: `any`
## Output types
- **`?`**
    - Comfy dtype: `*`
    - The result of the executed path, which could be from either the 'when_TRUE' or 'when_FALSE' branch, depending on the evaluation of the 'if_condition'. This output type name is represented by a placeholder '?', indicating the flexibility in the output based on conditional logic.
    - Python dtype: `any`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IfExecute:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "if_condition": (any,),
                "when_TRUE": (any,),
                "when_FALSE": (any,),
            },
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = "?"
    FUNCTION = "if_execute"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def if_execute(self, if_condition, when_TRUE, when_FALSE):
        return (when_TRUE if if_condition else when_FALSE,)

```
