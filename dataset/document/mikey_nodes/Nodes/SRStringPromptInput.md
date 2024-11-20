# SR String Prompt Input (Mikey)
## Documentation
- Class name: `SRStringPromptInput`
- Category: `Mikey/Meta`
- Output node: `False`

The SRStringPromptInput node is designed to integrate string inputs into a prompt's structure, specifically targeting the enhancement of semantic resolution within the prompt. It focuses on accepting a string input and embedding it into a given prompt's context, thereby enriching the prompt's descriptive capacity and specificity.
## Input types
### Required
- **`input_str`**
    - The primary string input that is to be integrated into the prompt. This input is crucial for tailoring the prompt's content to specific requirements or contexts, thereby enhancing its relevance and effectiveness.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The modified prompt structure, now containing the integrated string input. This output reflects the enhanced descriptive capacity of the prompt, achieved through the incorporation of the specified string.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SRStringPromptInput:
    @classmethod
    def INPUT_TYPES(s):
        return {'required': {'input_str': ('STRING', {'forceInput': True}),},
                "hidden": {"unique_id": "UNIQUE_ID", "prompt": "PROMPT"}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "add"
    CATEGORY = "Mikey/Meta"

    def add(self, input_str, unique_id=None, prompt=None):
        prompt.get(str(unique_id))['inputs']['sr_val'] = input_str
        return (input_str,)

```
