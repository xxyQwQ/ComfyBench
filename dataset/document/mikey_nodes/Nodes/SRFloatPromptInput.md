# SR Float Prompt Input (Mikey)
## Documentation
- Class name: `SRFloatPromptInput`
- Category: `Mikey/Meta`
- Output node: `False`

The SRFloatPromptInput node is designed for integrating floating-point values into a structured prompt system. It specifically handles the addition of floating-point input values to a prompt's metadata, facilitating dynamic content generation based on numerical input.
## Input types
### Required
- **`input_float`**
    - The primary floating-point value to be added to the prompt's metadata. It plays a crucial role in customizing the prompt content based on numerical input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`float`**
    - Comfy dtype: `FLOAT`
    - The same floating-point value that was input, indicating successful integration into the prompt.
    - Python dtype: `float`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SRFloatPromptInput:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_float': ('FLOAT', {'forceInput': True}),},
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "add"
    CATEGORY = "Mikey/Meta"

    def add(self, input_float, unique_id=None, prompt=None):
        prompt.get(str(unique_id))['inputs']['sr_val'] = str(input_float)
        return (input_float,)

```
