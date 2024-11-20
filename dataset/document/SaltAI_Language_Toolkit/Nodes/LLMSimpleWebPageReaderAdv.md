# ∞ Simple Web Page (Advanced)
## Documentation
- Class name: `LLMSimpleWebPageReaderAdv`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMSimpleWebPageReaderAdv node is designed to fetch and process web pages from a list of URLs, converting them into a structured document format. It optionally converts HTML content to text, facilitating the extraction of readable content from web pages for further analysis or processing.
## Input types
### Required
- **`urls`**
    - A list of URLs from which web pages will be fetched. This parameter is essential for the node's operation as it defines the sources of the web content to be processed.
    - Comfy dtype: `LIST`
    - Python dtype: `List[str]`
### Optional
- **`html_to_text`**
    - A boolean flag indicating whether the HTML content should be converted to plain text. This affects how the web page content is processed and presented in the output document.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The processed web page content, structured as documents. This output is useful for downstream text analysis or content management tasks.
    - Python dtype: `List[Document]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSimpleWebPageReaderAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "urls": ("LIST", {}),
            },
            "optional": {
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_web(self, urls, html_to_text=True):

        if not urls:
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReaderAdv")

        valid_urls = []
        for url in urls:
            if not valid_url(url):
                logger.warning("Skipping invalid URL", url)
                continue
            valid_urls.append(url)

        logger.info("Valided URLs:", valid_urls)

        documents = SimpleWebPageReader(html_to_text=html_to_text).load_data(valid_urls)
        return (documents,)

```
