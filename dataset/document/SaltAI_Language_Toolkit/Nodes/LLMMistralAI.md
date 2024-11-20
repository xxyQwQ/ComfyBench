# ∞ MistralAI Model
## Documentation
- Class name: `LLMMistralAI`
- Category: `SALT/Language Toolkit/Loaders`
- Output node: `False`

The LLMMistralAI node is designed to interface with the MistralAI API, allowing users to load and interact with various pre-trained language models provided by MistralAI. It facilitates the integration of advanced natural language processing capabilities into applications by leveraging the power of large language models.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the MistralAI model to be loaded. This parameter is crucial for determining which specific pre-trained model is utilized for processing, affecting the node's behavior and the nature of its output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`api_key`**
    - The API key required for authenticating with the MistralAI service. This parameter is essential for enabling access to the MistralAI models, ensuring secure and authorized use of the API.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `LLM_MODEL`
    - The output includes the loaded MistralAI model along with an embedding model, providing the necessary components for further natural language processing tasks.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMMistralAI:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/mistralai/
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "open-mistral-7b",
                    "open-mixtral-8x7b",
                    "mistral-small-latest",
                    "mistral-medium-latest",
                    "mistral-large-latest",
                ],),
                "api_key": ("STRING", {
                    "multiline": False, 
                    "default": os.environ.get("MISTRAL_API_KEY", "")
                }),
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name:str, api_key:str) -> Dict[str, Any]:
        if LAST_TOKENIZER:
            Settings.tokenizer = LAST_TOKENIZER
        llm = MistralAI(model_name=model_name, api_key=api_key)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "llm_name": model_name, "embed_model": embed_model, "embed_name": "bge-small-en-v1.5"},)

```
