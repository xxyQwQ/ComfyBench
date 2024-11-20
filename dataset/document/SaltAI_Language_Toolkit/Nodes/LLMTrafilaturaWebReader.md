# ∞ Trafilatura Web
## Documentation
- Class name: `LLMTrafilaturaWebReader`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMTrafilaturaWebReader node is designed to extract content from web pages using the Trafilatura library. It accepts up to four URLs as input and returns the extracted content as documents, facilitating the processing of web-based information for further analysis or consumption.
## Input types
### Required
- **`url_i`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
### Optional
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The extracted content from the specified web pages, returned as documents for easy integration into downstream processes.
    - Python dtype: `tuple[Document]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMTrafilaturaWebReader:
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
            },
        }

    RETURN_TYPES = ("DOCUMENT", )
    RETURN_NAMES = ("documents",)

    FUNCTION = "read_web_trafilatura"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_web_trafilatura(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMTrafilaturaWebReader")

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

        documents = TrafilaturaWebReader().load_data(valid_urls)
        return (documents,)

```
