---
tags:
- Prompt
---

# PromptLine
## Documentation
- Class name: `easy promptLine`
- Category: `EasyUse/Prompt`
- Output node: `False`

The `easy promptLine` node is designed to facilitate the manipulation and conditioning of textual prompts for generative models. It enables the dynamic integration and adjustment of negative conditioning phrases, ensuring that the generated content aligns with specified constraints and preferences. This node plays a crucial role in refining the input prompts to achieve desired outcomes in generative tasks, such as text generation or image synthesis, by allowing for precise control over the conditioning process.
## Input types
### Required
- **`prompt`**
    - Represents the primary textual content that the user inputs for processing. It serves as the base for any manipulations, additions, or exclusions applied by the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`start_index`**
    - Specifies the starting index from which the node begins to process the input prompt, allowing for segmented or partial processing of the text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_rows`**
    - Defines the maximum number of rows or lines to be processed or generated from the input prompt, setting a limit on the output's extent.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`STRING`**
    - Comfy dtype: `STRING`
    - The processed text output, which may include modifications, additions, or exclusions based on the node's functionality.
    - Python dtype: `str`
- **`COMBO`**
    - Comfy dtype: `*`
    - A combination of processed text outputs, potentially including multiple variations or manipulations of the input prompt.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class promptLine:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "text"}),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                     "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
                    },
            "hidden":{
                "workflow_prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING", AlwaysEqualProxy('*'))
    RETURN_NAMES = ("STRING", "COMBO")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_strings"
    CATEGORY = "EasyUse/Prompt"

    def generate_strings(self, prompt, start_index, max_rows, workflow_prompt=None, my_unique_id=None):
        lines = prompt.split('\n')
        lines = [zh_to_en([v])[0] if has_chinese(v) else v for v in lines if v]

        start_index = max(0, min(start_index, len(lines) - 1))

        end_index = min(start_index + max_rows, len(lines))

        rows = lines[start_index:end_index]

        return (rows, rows)

```
