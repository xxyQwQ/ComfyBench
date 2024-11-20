# ∞ Vector Store Index
## Documentation
- Class name: `LLMVectorStoreIndex`
- Category: `SALT/Language Toolkit/Indexing`
- Output node: `False`

This node is designed to create and manage a vector store index for language models, facilitating efficient storage, retrieval, and manipulation of vectorized representations of text data. It abstracts the complexities involved in handling large-scale vector data, optimizing for performance and scalability.
## Input types
### Required
- **`llm_model`**
    - Specifies the language model to be used for generating vector representations. It is crucial for determining the embedding model that will vectorize the text data.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `dict`
- **`document`**
    - The text document or a collection of documents to be indexed. This input is essential for generating the vector representations that will be stored in the index.
    - Comfy dtype: `DOCUMENT`
    - Python dtype: `str or list of str`
### Optional
- **`optional_llm_context`**
    - An optional context parameter that can be used to provide additional information or settings to the language model during the indexing process.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `dict`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The generated vector store index, which can be used for subsequent retrieval and manipulation of the vectorized text data.
    - Python dtype: `VectorStoreIndex`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMVectorStoreIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_MODEL",),
                "document": ("DOCUMENT",),
            },
            "optional": {
                "optional_llm_context": ("LLM_CONTEXT",),
            },
        }

    RETURN_TYPES = ("LLM_INDEX",)
    RETURN_NAMES = ("llm_index",)

    FUNCTION = "index"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Indexing"

    def index(self, llm_model, document, optional_llm_context = None):
        
        #document = cast(Sequence[Document], document) # This could be why documents are not working correctly
        embed_model = llm_model.get("embed_model", None)

        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")
        
        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
        tokenizer = MockTokenizer(max_tokens=1024, char_per_token=1)

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
