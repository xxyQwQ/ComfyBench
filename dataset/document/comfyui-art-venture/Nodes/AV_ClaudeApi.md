---
tags:
- LLM
- LLMChat
---

# Claude API
## Documentation
- Class name: `AV_ClaudeApi`
- Category: `ArtVenture/LLM`
- Output node: `False`

The AV_ClaudeApi node is designed to facilitate interaction with Claude's API, enabling the creation of API instances for further operations with Claude's language models. It abstracts the complexities of API key management and endpoint configuration, providing a streamlined way to access Claude's advanced language processing capabilities.
## Input types
### Required
- **`claude_api_key`**
    - The API key for Claude, essential for authenticating requests to the API. It's a critical component for enabling access to Claude's language models.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`endpoint`**
    - The URL of the Claude API endpoint. This parameter allows for specifying the API's location, with a default value pointing to Anthropic's API.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`version`**
    - The version of the Claude API to be used, allowing for control over which features and improvements are accessible.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`llm_api`**
    - Comfy dtype: `LLM_API`
    - The created instance of ClaudeApi, ready for making requests to Claude's language model API.
    - Python dtype: `ClaudeApi`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ClaudeApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "claude_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.anthropic.com/v1"}),
                "version": (["2023-06-01"], {"default": "2023-06-01"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    RETURN_NAMES = ("llm_api",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, claude_api_key, endpoint, version):
        if not claude_api_key or claude_api_key == "":
            claude_api_key = os.environ.get("CLAUDE_API_KEY")
        if not claude_api_key:
            raise Exception("Claude API key is required.")

        return (ClaudeApi(api_key=claude_api_key, endpoint=endpoint, version=version),)

```
