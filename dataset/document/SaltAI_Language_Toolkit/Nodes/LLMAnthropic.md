---
tags:
- LLM
- LLMChat
---

# ∞ Anthropic Model
## Documentation
- Class name: `LLMAnthropic`
- Category: `SALT/Language Toolkit/Loaders`
- Output node: `False`

The LLMAnthropic node integrates Anthropic's conversational AI models into the SaltAI Language Toolkit, enabling users to leverage advanced natural language processing capabilities for generating human-like text responses. This node is designed to facilitate seamless interaction with Anthropic's AI, providing a bridge for developers to incorporate sophisticated conversational AI features into their applications.
## Input types
### Required
- **`model`**
    - Specifies the Anthropic model to be used for generating responses. This selection determines the conversational capabilities and nuances of the AI's output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`anthropic_api_key`**
    - The API key required for accessing Anthropic's models. This key is essential for authenticating and enabling the interaction with Anthropic's AI services.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`embedding_model`**
    - Defines the embedding model to be used in conjunction with the main model. This affects the quality and relevance of the AI's responses by enhancing its understanding of the input context.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`openai_api_key`**
    - The API key for OpenAI, used when the embedding model requires access to OpenAI's services. This key facilitates the integration with OpenAI's ecosystem, broadening the AI's capabilities.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_model`**
    - Comfy dtype: `LLM_MODEL`
    - The loaded Anthropic model, ready for generating responses. This output signifies the successful initialization and readiness of the AI model for interaction.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMAnthropic:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/anthropic/
    """
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "model": (["claude-2.1", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],),
                "anthropic_api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("ANTHROPIC_API_KEY", "SALTAI_ANTHROPIC_KEY")
                }),
                "embedding_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-ada-002"},
                ),
                "openai_api_key": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": os.environ.get("OPENAI_API_KEY", "")
                }),
            },
        }

        if HAS_MULTI_ANTHROPIC:
            inputs.update({
                "optional": {
                    "multimodal": ("BOOLEAN", {"default": False})
                }
            })

        return inputs

    RETURN_TYPES = ("LLM_MODEL",)
    RETURN_NAMES = ("llm_model",)

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model: str, anthropic_api_key: str, embedding_model: str, openai_api_key: str, multimodal: bool = False) -> Dict[str, Any]:
        global LAST_TOKENIZER

        tokenizer = Anthropic().tokenizer
        LAST_TOKENIZER = Settings.tokenizer
        Settings.tokenizer = tokenizer

        if multimodal and HAS_MULTI_ANTHROPIC:
            llm = AnthropicMultiModal(model=model, api_key=anthropic_api_key, max_tokens=4096)
        else:
            llm = Anthropic(model=model, api_key=anthropic_api_key, max_tokens=4096)

        embed_model = OpenAIEmbedding(model_name=embedding_model, api_key=openai_api_key, max_tokens=4096)

        return ({"llm": llm, "llm_name": model, "embed_model": embed_model, "embed_name": embedding_model}, )

```
