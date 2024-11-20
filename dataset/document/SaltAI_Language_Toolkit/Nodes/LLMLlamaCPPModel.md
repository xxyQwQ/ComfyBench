---
tags:
- LoRA
---

# ∞ LlamaCPP Model
## Documentation
- Class name: `LLMLlamaCPPModel`
- Category: `SALT/Language Toolkit/Loaders`
- Output node: `False`

The LLMLlamaCPPModel node is designed to load and initialize LlamaCPP models, incorporating an embedding model alongside. It facilitates the integration of LlamaCPP models into larger workflows by providing a streamlined mechanism for model loading, including setting up necessary configurations and embedding models for enhanced functionality.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the LlamaCPP model to be loaded. This parameter is crucial for identifying the correct model file and initializing the LlamaCPP model accordingly.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `LLM_MODEL`
    - Returns a dictionary containing the loaded LlamaCPP model, its name, the associated embedding model, and the embedding model's name. This output is essential for subsequent processing or analysis steps in the workflow.
    - Python dtype: `Dict[str, Union[LlamaCPP, HuggingFaceEmbedding]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMLlamaCPP:
    """
    @Documentation: https://docs.llamaindex.ai/en/stable/api_reference/llms/llama_cpp/
    """
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (LOCAL_FILES, ), 
            },
        }

    RETURN_TYPES = ("LLM_MODEL", )
    RETURN_NAMES = ("model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Loaders"

    def load_model(self, model_name: str) -> Dict[str, Any]:
        if LAST_TOKENIZER:
            Settings.tokenizer = LAST_TOKENIZER
        path = folder_paths.get_full_path('llm', model_name)
        llm = LlamaCPP(model_path=path)
        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        return ({"llm":llm, "llm_name": model_name,"embed_model": embed_model, "embed_name": "BAAI/bge-small-en-v1.5"},)

```
