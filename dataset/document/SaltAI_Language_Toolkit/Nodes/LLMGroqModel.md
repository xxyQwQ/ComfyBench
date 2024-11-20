# ∞ Groq Model
## Documentation
- Class name: `LLMGroqModel`
- Category: `SALT/Language Toolkit/Loaders`
- Output node: `False`

The LLMGroqModel node integrates Groq's large language models (LLMs) with embedding models, such as those provided by OpenAI, to facilitate advanced text processing and analysis tasks. It enables the loading of specific LLMs and embedding models, configuring them for use in various applications that require understanding, generating, or transforming text.
## Input types
### Required
- **`model`**
    - Specifies the Groq model to be loaded for text processing tasks, indicating the specific LLM to be utilized.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`groq_api_key`**
    - The API key required to authenticate and access Groq's large language models, ensuring secure and authorized use.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`embedding_model`**
    - Defines the embedding model to be used in conjunction with the Groq LLM, typically for tasks involving text embedding or similarity analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`openai_api_key`**
    - Optional API key for accessing OpenAI's embedding models, providing flexibility in choosing the embedding model based on availability or preference.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`llm_model`**
    - Comfy dtype: `LLM_MODEL`
    - The loaded Groq large language model, ready for text processing tasks.
    - Python dtype: `dict`
- **`embed_model_only`**
    - Comfy dtype: `LLM_EMBED_MODEL`
    - The embedding model loaded alongside the Groq LLM, used for text embedding or similarity analysis.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMGroq:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/groq/
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],),
                "groq_api_key": ("STRING", {
                    "multiline": False,
                    "default": os.environ.get("GROQ_API_KEY", "SALTAI_GROQ_KEY")
                }),
                "embedding_model": (
                    sorted([x.value for x in OpenAIEmbeddingModelType]),
                    {"default": "text-embedding-ada-002"},
                ),
            },
            "optional": {
                "openai_api_key": ("STRING", {
                    "multiline": False,
                    "dynamicPrompts": False,
                    "default": os.environ.get("OPENAI_API_KEY", "SALTAI_OPENAI_KEY")
                }),
            }
        }

    RETURN_TYPES = ("LLM_MODEL", "LLM_EMBED_MODEL")
    RETURN_NAMES = ("llm_model", "embed_model_only")

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model: str, groq_api_key: str, embedding_model:str, openai_api_key:str = None) -> Dict[str, Any]:
        if LAST_TOKENIZER:
            Settings.tokenizer = LAST_TOKENIZER
        llm = Groq(model=model, api_key=groq_api_key)
        embed_model = OpenAIEmbedding(model_name=embedding_model, api_key=openai_api_key,)
        return ({"llm": llm, "llm_name": model, "embed_model": embed_model, "embed_name": embedding_model}, {"embed_model": embed_model, "embed_name": embedding_model})

```
