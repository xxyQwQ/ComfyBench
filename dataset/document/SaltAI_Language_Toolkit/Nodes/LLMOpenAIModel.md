---
tags:
- LoRA
---

# ∞ OpenAI Model
## Documentation
- Class name: `LLMOpenAIModel`
- Category: `SALT/Language Toolkit/Loaders`
- Output node: `False`

The LLMOpenAIModel node is designed to interface with OpenAI's language models, facilitating the loading and utilization of these models for various language processing tasks. It abstracts the complexities involved in interacting with OpenAI's API, providing a streamlined way to leverage their advanced natural language understanding and generation capabilities.
## Input types
### Required
- **`model`**
    - Specifies the name of the OpenAI model to be loaded. This parameter is crucial for determining which specific language model is utilized for processing, affecting the node's behavior and the quality of its outputs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`api_key`**
    - The API key required to authenticate with OpenAI's services. This key enables the node to access OpenAI's language models, making it essential for the operation of the node.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`embedding_model`**
    - Defines the model used for embedding purposes, which is part of OpenAI's suite. This parameter influences how text inputs are converted into embeddings, affecting tasks such as similarity searches or contextual analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`multimodal`**
    - A boolean flag indicating whether the multimodal capabilities of the model should be enabled. This affects whether the model can process and understand inputs that include both text and other modalities, such as images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`llm_model`**
    - Comfy dtype: `LLM_MODEL`
    - The loaded language model from OpenAI, ready for language processing tasks.
    - Python dtype: `Dict[str, Any]`
- **`embed_model_only`**
    - Comfy dtype: `LLM_EMBED_MODEL`
    - The embedding model loaded alongside the main language model, used for generating embeddings from text inputs.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMOpenAI:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/openai/
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (list(ALL_AVAILABLE_MODELS),),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("OPENAI_API_KEY", "SALTAI_OPENAI_KEY")
                }),
                "embedding_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-ada-002"},
                ),
            },
            "optional": {
                "multimodal": ("BOOLEAN", {"default": False})
            }
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_EMBED_MODEL")
    RETURN_NAMES = ("llm_model", "embed_model_only")

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model:str, embedding_model:str, api_key:str, multimodal:bool = False) -> Dict[str, Any]:
        if LAST_TOKENIZER:
            Settings.tokenizer = LAST_TOKENIZER
        llm = OpenAI(model=model, api_key=api_key) if not multimodal else OpenAIMultiModal(model=model, api_key=api_key)
        embed_model = OpenAIEmbedding(model_name=embedding_model, api_key=api_key,)
        return ({"llm":llm, "llm_name": model, "embed_model":embed_model, "embed_name": embedding_model}, {"embed_model": embed_model, "embed_name": embedding_model})

```
