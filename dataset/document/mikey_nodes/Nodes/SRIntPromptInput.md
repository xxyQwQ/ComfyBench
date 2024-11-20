# SR Int Prompt Input (Mikey)
## Documentation
- Class name: `SRIntPromptInput`
- Category: `Mikey/Meta`
- Output node: `False`

The SRIntPromptInput node is designed to integrate an integer input into a structured prompt, enhancing it with specific numerical data. This node plays a crucial role in customizing prompts with integer values, thereby enabling more precise and context-aware generation processes.
## Input types
### Required
- **`input_int`**
    - The primary integer input required for the operation. It directly influences the customization of the prompt by adding a numerical value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`output_int`**
    - Comfy dtype: `INT`
    - The integer input after being processed and integrated into the prompt, reflecting the customization applied.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SRIntPromptInput:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_int': ('INT', {'forceInput': True}),},
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("output_int",)
    FUNCTION = "add"
    CATEGORY = "Mikey/Meta"

    def add(self, input_int, extra_pnginfo, unique_id, prompt):
        prompt.get(str(unique_id))['inputs']['sr_val'] = str(input_int)
        return (input_int,)

```
