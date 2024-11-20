# ∞ Query Engine
## Documentation
- Class name: `LLMQueryEngine`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

The LLMQueryEngine node is designed to facilitate querying large language models (LLMs) by assembling and sending structured queries based on user input and predefined message components. It abstracts the complexity of interacting with LLMs, enabling users to obtain responses to queries efficiently.
## Input types
### Required
- **`llm_model`**
    - Specifies the large language model to be used for the query, including any necessary model configurations.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`llm_index`**
    - Represents the index or interface through which the query will be executed, potentially encapsulating specific querying mechanisms or optimizations.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `SpecificTypeRepresentingLLMIndex`
### Optional
- **`query`**
    - The user's query input, which will be included in the final assembled query sent to the LLM.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`llm_message`**
    - A list of pre-defined messages or prompts that can be included in the query to guide the LLM's response generation.
    - Comfy dtype: `LIST`
    - Python dtype: `Optional[List[str]]`
## Output types
- **`results`**
    - Comfy dtype: `STRING`
    - The output from the LLM in response to the assembled query.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMQueryEngine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL", ),
                "llm_index": ("LLM_INDEX", ),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def query_engine(self, llm_model, llm_index, query=None, llm_message=None):
        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg).strip():
                    query_components.append(str(msg))
        else:
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

        if query:
            if query.strip():
                query_components.append("user: " + query)
        query_components.append("assistant:")

        query_join = "\n".join(query_components)

        query_engine = llm_index.as_query_engine(llm=llm_model.get("llm", None), embed_model=llm_model.get("embed_model", None))
        response = query_engine.query(query_join)
        return (response.response,)

```
