---
tags:
- LLM
- LLMChat
---

# OpenAI API
## Documentation
- Class name: `AV_OpenAIApi`
- Category: `ArtVenture/LLM`
- Output node: `False`

The AV_OpenAIApi node facilitates the creation of an API client for interacting with OpenAI's services, enabling the integration of OpenAI's language model capabilities within the ArtVenture ecosystem. It abstracts the authentication and setup process, making it easier to leverage OpenAI's API for various language processing tasks.
## Input types
### Required
- **`openai_api_key`**
    - The API key for authenticating requests to OpenAI's services. It is essential for enabling secure communication with OpenAI's API.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`endpoint`**
    - The URL endpoint for OpenAI's API. It defaults to OpenAI's official API endpoint but can be customized if needed.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_api`**
    - Comfy dtype: `LLM_API`
    - Returns an instance of the OpenAIApi, configured and ready to interact with OpenAI's language models.
    - Python dtype: `OpenAIApi`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OpenAIApiNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "openai_api_key": ("STRING", {"multiline": False}),
                "endpoint": ("STRING", {"multiline": False, "default": "https://api.openai.com/v1"}),
            },
        }

    RETURN_TYPES = ("LLM_API",)
    FUNCTION = "create_api"
    CATEGORY = "ArtVenture/LLM"

    def create_api(self, openai_api_key, endpoint):
        if not openai_api_key or openai_api_key == "":
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            raise Exception("OpenAI API key is required.")

        return (OpenAIApi(api_key=openai_api_key, endpoint=endpoint),)

```
