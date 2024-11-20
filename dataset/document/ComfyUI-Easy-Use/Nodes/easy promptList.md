---
tags:
- Prompt
---

# PromptList
## Documentation
- Class name: `easy promptList`
- Category: `EasyUse/Prompt`
- Output node: `False`

This node is designed to facilitate the generation of string lists from multiline prompts, with enhanced capabilities for processing prompts containing Chinese characters by converting them to English. It uniquely caters to workflows requiring specific starting points and row limitations in the output list, while also accommodating unique identifiers.
## Input types
### Required
- **`prompt_i`**
    - Represents a main text input from which the list of strings will be generated. Supports multiline input and is capable of handling multiple prompts sequentially numbered (e.g., prompt_1, prompt_2, etc.).
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`optional_prompt_list`**
    - An optional list parameter that allows for additional prompts to be included in the generation process.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
## Output types
- **`prompt_list`**
    - Comfy dtype: `LIST`
    - The generated list of strings derived from the input prompts.
    - Python dtype: `List[str]`
- **`prompt_strings`**
    - Comfy dtype: `STRING`
    - A concatenated string of all input prompts, separated by a specified delimiter.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class promptList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt_1": ("STRING", {"multiline": True, "default": ""}),
            "prompt_2": ("STRING", {"multiline": True, "default": ""}),
            "prompt_3": ("STRING", {"multiline": True, "default": ""}),
            "prompt_4": ("STRING", {"multiline": True, "default": ""}),
            "prompt_5": ("STRING", {"multiline": True, "default": ""}),
        },
            "optional": {
                "optional_prompt_list": ("LIST",)
            }
        }

    RETURN_TYPES = ("LIST", "STRING")
    RETURN_NAMES = ("prompt_list", "prompt_strings")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "run"
    CATEGORY = "EasyUse/Prompt"

    def run(self, **kwargs):
        prompts = []

        if "optional_prompt_list" in kwargs:
            for l in kwargs["optional_prompt_list"]:
                prompts.append(l)

        # Iterate over the received inputs in sorted order.
        for k in sorted(kwargs.keys()):
            v = kwargs[k]

            # Only process string input ports.
            if isinstance(v, str) and v != '':
                if has_chinese(v):
                    v = zh_to_en([v])[0]
                prompts.append(v)

        return (prompts, prompts)

```
