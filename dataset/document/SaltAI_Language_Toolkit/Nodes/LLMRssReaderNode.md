# ∞ RSS
## Documentation
- Class name: `LLMRssReaderNode`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMRssReaderNode is designed to fetch and parse RSS feeds from provided URLs, converting the feed data into a structured document format. This node facilitates the aggregation of content from multiple RSS sources, streamlining the process of accessing and analyzing news or blog feeds programmatically.
## Input types
### Required
- **`url_i`**
    - The URL from which to read the RSS feed. This parameter is mandatory and initiates the RSS reading process. Up to four URLs can be provided, allowing for the aggregation of content from multiple sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The structured document format output containing the aggregated content from the RSS feeds.
    - Python dtype: `tuple`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMRssReaderNode:
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

    FUNCTION = "read_rss"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def read_rss(self, url_1, url_2=None, url_3=None, url_4=None):
        if not url_1.strip():
            raise ValueError("At least one URL must be provided to LLMSimpleWebPageReader")

        urls = [url_1.strip()]
        if url_2.strip():
            urls.append(url_2.strip())
        if url_3.strip():
            urls.append(url_3.strip())
        if url_4.strip():
            urls.append(url_4.strip())

        urls = [url for url in urls if valid_url(url)]

        logger.info("Valided URLs:", urls)

        documents = RssReader().load_data(urls)
        return (documents,)

```
