# LayerUtility: String Condition
## Documentation
- Class name: `LayerUtility: StringCondition`
- Category: `😺dzNodes/LayerUtility/Data`
- Output node: `False`

The StringCondition node is designed to evaluate text against specified conditions, such as inclusion or exclusion of a substring, and return a boolean result along with the result in string format. This functionality allows for conditional logic to be applied based on text content, enabling dynamic flow control and content filtering within a data processing pipeline.
## Input types
### Required
- **`text`**
    - The 'text' parameter is the primary string to be evaluated against the given condition. Its role is crucial as it determines the context in which the condition is applied, directly influencing the outcome of the node's execution.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`condition`**
    - The 'condition' parameter specifies the type of string evaluation to perform, such as 'include' or 'exclude', dictating whether the substring should be present or absent in the text for the condition to be considered true.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sub_string`**
    - The 'sub_string' parameter is the string fragment to be searched within the 'text' parameter. Its presence or absence, based on the 'condition', determines the boolean outcome of the node's operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`output`**
    - Comfy dtype: `BOOLEAN`
    - Indicates whether the specified condition (inclusion or exclusion of a substring) was met in the text.
    - Python dtype: `bool`
- **`string`**
    - Comfy dtype: `STRING`
    - A string representation of the boolean result, providing a textual indication of whether the condition was met.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class StringCondition:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(self):
        string_condition_list = ["include", "exclude",]
        return {"required": {
                "text": ("STRING", {"multiline": False}),
                "condition": (string_condition_list,),
                "sub_string": ("STRING", {"multiline": False}),
            },}

    RETURN_TYPES = ("BOOLEAN", "STRING",)
    RETURN_NAMES = ("output", "string",)
    FUNCTION = 'string_condition'
    CATEGORY = '😺dzNodes/LayerUtility/Data'

    def string_condition(self, text, condition, sub_string):
        ret = False
        if condition == "include":
            ret = sub_string in text
        if condition == "exclude":
            ret = sub_string not in text
        return (ret, str(ret))

```
