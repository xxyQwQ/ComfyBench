# LayerUtility: Switch Case
## Documentation
- Class name: `LayerUtility: SwitchCase`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The SwitchCase node provides a mechanism to execute different branches of logic based on the value of a given condition. It compares a switch condition against multiple cases and executes the branch where the condition matches, allowing for more dynamic and conditional data flow within a node network.
## Input types
### Required
- **`switch_condition`**
    - The condition value to be compared against the defined cases. It determines which branch of logic will be executed based on its match with the case values.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`case_i`**
    - A case value to compare against the switch condition. If matched, the corresponding input for this case is selected for output. The index 'i' can range to accommodate multiple cases.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`input_default`**
    - The default input to use if none of the case values match the switch condition.
    - Comfy dtype: `*`
    - Python dtype: `any`
### Optional
- **`input_i`**
    - The input associated with a case. It is selected for output if the switch condition matches the corresponding case value. The index 'i' can range to accommodate multiple inputs.
    - Comfy dtype: `*`
    - Python dtype: `any`
## Output types
- **`?`**
    - Comfy dtype: `*`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SwitchCaseNode:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "switch_condition": ("STRING", {"default": "", "multiline": False}),
                "case_1": ("STRING", {"default": "", "multiline": False}),
                "case_2": ("STRING", {"default": "", "multiline": False}),
                "case_3": ("STRING", {"default": "", "multiline": False}),
                "input_default": (any,),
            },
            "optional": {
                "input_1": (any,),
                "input_2": (any,),
                "input_3": (any,),
            }
        }

    RETURN_TYPES = (any,)
    RETURN_NAMES = "?"
    FUNCTION = "switch_case"
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def switch_case(self, switch_condition, case_1, case_2, case_3, input_default, input_1=None, input_2=None, input_3=None):

        output=input_default
        if switch_condition == case_1 and input_1 is not None:
            output=input_1
        elif switch_condition == case_2 and input_2 is not None:
            output=input_2
        elif switch_condition == case_3 and input_3 is not None:
            output=input_3

        return (output,)

```
