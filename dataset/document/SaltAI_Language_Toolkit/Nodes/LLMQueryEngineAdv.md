---
tags:
- LLM
---

# ∞ Query Engine (Advanced)
## Documentation
- Class name: `LLMQueryEngineAdv`
- Category: `SALT/Language Toolkit/Querying`
- Output node: `False`

The LLMQueryEngineAdv node is designed to leverage language models for advanced querying capabilities. It constructs a comprehensive query from user inputs and messages, utilizes an embedding model for query expansion, and employs a vector index retriever with post-processing for similarity-based filtering. This node aims to provide precise and relevant responses by integrating language understanding and retrieval technologies.
## Input types
### Required
- **`llm_model`**
    - The language model and optional embedding model used for query expansion and understanding. It's crucial for interpreting the query and retrieving relevant information.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`llm_index`**
    - The index used for retrieving query results based on vector similarity. Essential for the node's ability to find the most relevant responses.
    - Comfy dtype: `LLM_INDEX`
    - Python dtype: `Any`
### Optional
- **`query`**
    - The user's query, which is combined with other messages to form the complete query input. It plays a key role in directing the search and retrieval process.
    - Comfy dtype: `STRING`
    - Python dtype: `Optional[str]`
- **`llm_message`**
    - A list of messages that can be included in the query construction. These messages add context and detail to the query, enhancing its comprehensiveness.
    - Comfy dtype: `LIST`
    - Python dtype: `Optional[List[Any]]`
- **`top_k`**
    - Specifies the number of top similar results to retrieve. It determines the breadth of the search results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`similarity_cutoff`**
    - The similarity threshold for filtering results. Only results with a similarity score above this cutoff are considered relevant.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`results`**
    - Comfy dtype: `STRING`
    - The retrieved and processed query response, which is expected to be relevant and precise based on the input query and similarity criteria.
    - Python dtype: `Tuple[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMQueryEngineAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "llm_index": ("LLM_INDEX",),
            },
            "optional": {
                "query": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Type your query here"}),
                "llm_message": ("LIST", {}),
                "top_k": ("INT", {"default": 10}),
                "similarity_cutoff": ("FLOAT", {"default": 0.7}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("results",)

    FUNCTION = "query_engine"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Querying"

    def query_engine(self, llm_model, llm_index, query=None, llm_message=None, top_k=10, similarity_cutoff=0.7):

        model = llm_model['llm']
        embed_model = llm_model.get('embed_model', None)

        Settings.llm = model
        Settings.embed_model = embed_model
        
        if not embed_model:
            raise AttributeError("Unable to determine embed model from provided `LLM_MODEL` input.")

        query_components = []
        
        if llm_message and isinstance(llm_message, list):
            for msg in llm_message:
                if str(msg.content).strip():
                    query_components.append(str(msg.content))
        else:
            query_components.append("Analyze the above document carefully to find your answer. If you can't find one, say so.")

        if query and query.strip():
            query_components.append("user: " + query)

        query_components.append("assistant:")

        query_join = "\n".join(query_components)

        retriever = VectorIndexRetriever(index=llm_index, similarity_top_k=top_k, embed_model=embed_model)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],
        )

        response = query_engine.query(query_join)

        Settings.llm = None
        Settings.embed_model = None

        return (response.response,)

```
