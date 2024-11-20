# ∞ Simple Web Page
## Documentation
- Class name: `LLMSimpleWebPageReader`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMSimpleWebPageReader node is designed to fetch and process web pages from specified URLs, converting them into a document format. It supports reading from multiple URLs, with the option to convert HTML content to text, facilitating the extraction of useful information from web pages.
## Input types
### Required
- **`url_i`**
    - The primary URL from which the web page will be read. This is a mandatory parameter to initiate the reading process. The index 'i' indicates that multiple URLs can be specified, allowing for the reading of multiple web pages in a single operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`html_to_text`**
    - A boolean flag indicating whether to convert HTML content to plain text, enhancing readability and processing of the web page content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The processed documents obtained from the web pages, ready for further analysis or storage.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSimpleWebPageReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_1": ("STRING", {}),
            },
            "optional": {
                "url_2": ("STRING", {}),
                "url_3": ("STRING", {}),
                "url_4": ("STRING", {}),
                "html_to_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_web(self, url_1, url_2=None, url_3=None, url_4=None, html_to_text=True):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")

        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())

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
