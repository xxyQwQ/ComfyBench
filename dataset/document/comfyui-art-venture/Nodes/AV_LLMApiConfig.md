---
tags:
- LLM
- LLMChat
---

# LLM API Config
## Documentation
- Class name: `AV_LLMApiConfig`
- Category: `ArtVenture/LLM`
- Output node: `False`

The AV_LLMApiConfig node is designed to generate configuration settings for various large language models (LLMs) by specifying model type, maximum token count, and temperature. This configuration is essential for tailoring the behavior of LLMs to specific tasks or preferences, providing a foundation for further interactions with these models.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for the LLM configuration. This parameter supports a wide range of models, including GPT, Claude, and Bedrock variants, allowing for flexible model selection based on the task at hand.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_token`**
    - Defines the maximum number of tokens the LLM can generate or process in a single request. This parameter helps control the length of the output, ensuring it meets specific requirements or limitations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`temperature`**
    - Controls the creativity or randomness of the LLM's responses. A higher temperature leads to more varied outputs, while a lower temperature results in more deterministic and predictable text.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`llm_config`**
    - Comfy dtype: `LLM_CONFIG`
    - The output is a configuration object tailored for LLM interactions, encapsulating the specified model, token limit, and temperature settings.
    - Python dtype: `LLMConfig`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMApiConfigNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    gpt_models
                    + claude3_models
                    + claude2_models
                    + bedrock_claude3_models
                    + bedrock_claude2_models
                    + bedrock_mistral_models,
                    {"default": gpt_vision_models[0]},
                ),
                "max_token": ("INT", {"default": 1024}),
                "temperature": ("FLOAT", {"default": 0, "min": 0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LLM_CONFIG",)
    RETURN_NAMES = ("llm_config",)
    FUNCTION = "make_config"
    CATEGORY = "ArtVenture/LLM"

    def make_config(self, max_token, model, temperature):
        return (LLMConfig(model=model, max_token=max_token, temperature=temperature),)

```
