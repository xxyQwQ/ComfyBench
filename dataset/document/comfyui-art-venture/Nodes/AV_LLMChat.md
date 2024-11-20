---
tags:
- LLM
- LLMChat
---

# LLM Chat
## Documentation
- Class name: `AV_LLMChat`
- Category: `ArtVenture/LLM`
- Output node: `False`

The LLMChat node facilitates conversation with language models by processing and forwarding messages, configurations, and optional seed information to an API for generating responses. It abstracts the complexities of interacting with different language model APIs, providing a unified interface for initiating chat sessions.
## Input types
### Required
- **`messages`**
    - A list of messages that include both user and system messages, which are processed and formatted for the API call. This parameter is crucial for determining the flow and context of the conversation.
    - Comfy dtype: `LLM_MESSAGE`
    - Python dtype: `List[LLMMessage]`
- **`api`**
    - The API object that specifies the language model to be used for the chat session. It plays a key role in directing the request to the appropriate language model API.
    - Comfy dtype: `LLM_API`
    - Python dtype: `LLMApi`
- **`config`**
    - Configuration settings for the language model, including model selection and parameters like max tokens and temperature, which influence the generation's verbosity and creativity.
    - Comfy dtype: `LLM_CONFIG`
    - Python dtype: `LLMConfig`
- **`seed`**
    - An optional seed for deterministic output, enhancing reproducibility of the chat responses.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`response`**
    - Comfy dtype: `STRING`
    - The generated text response from the language model, encapsulating the model's reply to the input messages.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "messages": ("LLM_MESSAGE",),
                "api": ("LLM_API",),
                "config": ("LLM_CONFIG",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x1FFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "chat"
    CATEGORY = "ArtVenture/LLM"

    def chat(self, messages: List[LLMMessage], api: LLMApi, config: LLMConfig, seed):
        response = api.chat(messages, config, seed)
        return (response,)

```
