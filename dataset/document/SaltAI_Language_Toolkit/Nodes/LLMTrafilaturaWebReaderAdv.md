---
tags:
- LLM
- LlamaIndex
- WebContentExtraction
---

# ∞ Trafilatura Web (Advanced)
## Documentation
- Class name: `LLMTrafilaturaWebReaderAdv`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMTrafilaturaWebReaderAdv node is designed to extract content from web pages provided as a list of URLs. It validates the URLs and uses the Trafilatura library to efficiently scrape and process web content, returning it in a structured document format.
## Input types
### Required
- **`urls`**
    - A list of URLs from which the content is to be extracted. This parameter is essential for the node's operation as it defines the sources of the web content to be processed.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The structured content extracted from the provided web pages, returned as documents.
    - Python dtype: `Tuple[List[Dict[str, Any]],]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMTrafilaturaWebReaderAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("LIST", {}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_web_trafilatura(self, urls):

        if not urls:
            raise ValueError("At least one URL must be provided to LLMTrafilaturaWebReaderAdv")

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                logger.warning("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        logger.info("Valided URLs:", valid_urls)

        documents = TrafilaturaWebReader().load_data(valid_urls)
        return (documents,)

```
