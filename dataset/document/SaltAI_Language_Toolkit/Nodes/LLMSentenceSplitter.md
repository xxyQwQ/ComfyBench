# ∞ Setence Splitter
## Documentation
- Class name: `LLMSentenceSplitter`
- Category: `SALT/Language Toolkit/Parsing`
- Output node: `False`

The LLMSentenceSplitter node is designed to segment text into smaller, manageable chunks based on specified size and overlap criteria. This functionality is crucial for processing large documents or texts in a way that makes them more amenable to detailed analysis or further processing steps.
## Input types
### Required
- **`chunk_size`**
    - Specifies the maximum size of each text chunk. It determines how large each segment of text will be, directly impacting the granularity of the splitting process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`chunk_overlap`**
    - Defines the number of characters that will overlap between consecutive text chunks. This overlap ensures continuity and context preservation across the segmented text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`llm_sentence_splitter`**
    - Comfy dtype: `LLM_SENTENCE_SPLITTER`
    - Produces an instance of a SentenceSplitter, configured with the provided chunk size and overlap, ready to be used for text segmentation.
    - Python dtype: `SentenceSplitter`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSentenceSplitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "chunk_size": ("INT", {"min": 8, "max": 2048, "step": 1, "default": 1024}),
                "chunk_overlap": ("INT", {"min": 0, "max": 2048, "step": 1, "default": 20})
            },
        }

    RETURN_TYPES = ("LLM_SENTENCE_SPLITTER",)
    RETURN_NAMES = ("llm_sentence_splitter",)

    FUNCTION = "semantic_nodes"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Parsing"

    def semantic_nodes(self, chunk_size, chunk_overlap):
        splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return (splitter, )

```
