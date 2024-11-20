# ∞ Summary Index
## Documentation
- Class name: `LLMSummaryIndex`
- Category: `SALT/Language Toolkit/Indexing`
- Output node: `False`

The LLMSummaryIndex node is designed to create a summary index from a collection of documents using a specified language model embedding. It processes each document to handle metadata and text, applying sentence splitting as part of its transformation steps, and then constructs an index that can be used for summarizing or retrieving information from the documents.
## Input types
### Required
- **`llm_model`**
    - The language model and its embedding model used for generating document summaries. It's crucial for determining the embedding strategy for the documents.
    - Comfy dtype: `LLM_MODEL`
    - Python dtype: `Dict[str, Any]`
- **`document`**
    - A list of documents to be indexed. Each document's text and metadata are processed for indexing.
    - Comfy dtype: `DOCUMENT`
    - Python dtype: `List[Document]`
### Optional
- **`optional_llm_context`**
    - Optional context provided to the language model during the indexing process. It can be used to tailor the embedding process to specific requirements or contexts.
    - Comfy dtype: `LLM_CONTEXT`
    - Python dtype: `Optional[Dict[str, Any]]`
## Output types
- **`llm_index`**
    - Comfy dtype: `LLM_INDEX`
    - The summary index created from the documents, ready for use in summarization or information retrieval tasks.
    - Python dtype: `SummaryIndex`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSummaryIndex:
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

    def index(self, llm_model, document, optional_llm_context=None):

        embed_model = llm_model.get("embed_model", None)
        
        if not embed_model:
            raise ValueError("Unable to determine LLM Embedding Model")

        splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=0)
        tokenizer = MockTokenizer(max_tokens=1024, char_per_token=1)

        documents = []
        for doc in document:
            logger.info("Document:")
            logger.data(doc)
            logger.info("\n==================\n")
            metadata = {}
            text = doc.text
            if doc.metadata:
                metadata = doc.metadata
                token_count = tokenizer.count(metadata)
                if token_count > 1024:
                    metadata = tokenizer.truncate(metadata)
            documents.append(Document(text=text, extra_info=metadata))

        index = SummaryIndex.from_documents(
            documents, 
            embed_model=embed_model, 
            service_context=optional_llm_context or None,
            transformations=[splitter]
            )
        return (index,)

```
