---
tags:
- LLM
- LLMChat
---

# LLM Completion
## Documentation
- Class name: `AV_LLMCompletion`
- Category: `ArtVenture/LLM`
- Output node: `False`

The AV_LLMCompletion node facilitates generating text completions using a language model. It processes a given text prompt through a specified API and configuration, optionally incorporating a seed for deterministic outputs.
## Input types
### Required
- **`prompt`**
    - The text prompt to be completed by the language model. It is a crucial input as it guides the generation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`api`**
    - The API used to interact with the language model, determining the specific model and its capabilities.
    - Comfy dtype: `LLM_API`
    - Python dtype: `LLMApi`
- **`config`**
    - Configuration settings for the language model, including parameters like model type and generation settings.
    - Comfy dtype: `LLM_CONFIG`
    - Python dtype: `LLMConfig`
- **`seed`**
    - An optional seed value for generating deterministic outputs, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`response`**
    - Comfy dtype: `STRING`
    - The generated text completion from the language model.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMCompletionNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "api": ("LLM_API",),
                "config": ("LLM_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x1FFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, prompt: str, api: LLMApi, config: LLMConfig, seed):
        response = api.complete(prompt, config, seed)
        return (response,)

```
