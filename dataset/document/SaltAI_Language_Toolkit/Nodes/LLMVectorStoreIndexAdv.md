---
tags:
- LLM
---

# ∞ Vector Store Index (Adv)
## Documentation
- Class name: `LLMVectorStoreIndexAdv`
- Category: `SALT/Language Toolkit/Indexing`
- Output node: `False`

The LLMVectorStoreIndexAdv node is designed for advanced vector storage indexing within a language model framework. It extends the capabilities of standard vector indexing by incorporating additional features or optimizations tailored for complex language processing tasks. This node is pivotal for enhancing the efficiency and accuracy of retrieving and managing vectorized representations of text data in large-scale language models.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used for indexing, serving as the core component for generating vector representations of text.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `str`
- **`document`**
    - The text document to be indexed, which is processed to generate its vector representation.
    - Comfy dtype: `DOCUMENT`
    - Python dtype: `str`
### Optional
- **`chunk_size`**
    - Defines the size of text chunks for processing, allowing for customization of indexing granularity.
    - Comfy dtype: `COMBO[INT]`
    - Python dtype: `List[int]`
- **`chunk_overlap`**
    - Determines the overlap between consecutive text chunks to ensure continuity in the indexing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_llm_context`**
    - Provides additional context from a language model to enhance the indexing accuracy.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `str`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The result of the indexing process, representing the vectorized form of the input document.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMVectorStoreIndexAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "document": ("DOCUMENT",),
            },
            "optional": {
                "chunk_size": ([128, 256, 512, 1024, 2048],),
                "chunk_overlap": ("INT", {"min": 0, "max": 512, "default": 0}),
                "optional_llm_context": ("LLM_CONTEXT",)
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Indexing"

    def index(self, llm_model, document, chunk_size=1024, chunk_overlap=0, optional_llm_context=None):
        
        embed_model = llm_model.get("embed_model", None)
        
        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        tokenizer = MockTokenizer(max_tokens=chunk_size, char_per_token=1)

        #documents = []
        #for doc in document:
        #    logger.info("Document:")
        #    logger.data(doc)
        #    logger.info("\n==================\n")
        #    metadata = {}
        #    text = doc.text
        #    if doc.metadata:
        #        metadata = doc.metadata
        #        token_count = tokenizer.count(metadata)
        #        if token_count > 1024:
        #            metadata = tokenizer.truncate(metadata)
        #    documents.append(Document(text=text, extra_info=metadata))
        
        index = VectorStoreIndex.from_documents(
            document, 
            embed_model=embed_model,
            service_context=optional_llm_context,
            transformations=[splitter]
        )
        
        return (index,)

```
