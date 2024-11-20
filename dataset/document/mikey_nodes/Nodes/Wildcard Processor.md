---
tags:
- Prompt
- Text
- Wildcard
---

# Wildcard Processor (Mikey)
## Documentation
- Class name: `Wildcard Processor`
- Category: `Mikey/Text`
- Output node: `False`

The Wildcard Processor node is designed to enhance text inputs by dynamically substituting placeholders with specified or random values. It leverages wildcard and random syntax processing to generate varied outputs based on the given seed, making it ideal for applications requiring text variation and customization.
## Input types
### Required
- **`prompt`**
    - The primary text input containing placeholders for dynamic substitution. It serves as the base for wildcard and random syntax processing, directly influencing the generated output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - A numerical value used to initialize the random number generator, ensuring the reproducibility of text variations. It plays a crucial role in the deterministic transformation of the input text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The processed text with wildcards and optional elements dynamically replaced, ready for further use or display.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [LMStudioPrompt](../../mikey_nodes/Nodes/LMStudioPrompt.md)
    - [Prompt With Style V3](../../mikey_nodes/Nodes/Prompt With Style V3.md)



## Source code
```python
class WildcardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"prompt": ("STRING", {"multiline": True, "placeholder": "Prompt Text"}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})},
                "hidden": {"prompt_": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},}

    RETURN_TYPES = ('STRING',)
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Text'

    def process(self, prompt, seed, prompt_=None, extra_pnginfo=None):
        if prompt_ is None:
            prompt_ = {}
        if extra_pnginfo is None:
            extra_pnginfo = {}
        prompt = search_and_replace(prompt, extra_pnginfo, prompt_)
        prompt = process_wildcard_syntax(prompt, seed)
        prompt = process_random_syntax(prompt, seed)
        new_prompt = find_and_replace_wildcards(prompt, seed)
        # loop to pick up wildcards that are in wildcard files
        if new_prompt != prompt:
            for i in range(10):
                prompt = new_prompt
                prompt = search_and_replace(prompt, extra_pnginfo, prompt_)
                prompt = process_wildcard_syntax(prompt, seed)
                prompt = process_random_syntax(prompt, seed)
                new_prompt = find_and_replace_wildcards(prompt, seed)
                if new_prompt == prompt:
                    break
        return (new_prompt, )

```
