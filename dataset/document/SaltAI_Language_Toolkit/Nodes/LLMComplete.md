# ∞ Complete Query
## Documentation
- Class name: `LLMComplete`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

The LLMComplete node is designed to generate text completions based on a given prompt, utilizing a specified large language model (LLM). It abstracts the complexity of interacting with different LLMs, providing a unified interface for text generation tasks.
## Input types
### Required
- **`llm_model`**
    - Specifies the large language model to be used for generating text completions. It is crucial for determining the source and capabilities of the generated text.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`prompt`**
    - The input text prompt based on which the text completion is generated. It sets the context or question for the LLM to respond to, influencing the nature of the generated text.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`completion`**
    - Comfy dtype: `STRING`
    - The generated text completion from the LLM based on the provided prompt. It represents the LLM's response or continuation of the input prompt.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMComplete:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "The circumference of the Earth is"}),
            },
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("completion", )

    FUNCTION = "complete"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def complete(self, llm_model:Dict[str, Any], prompt:str) -> str:
        response = llm_model['llm'].complete(prompt)
        return (response.text, )

```
