# ∞ Jina.AI Scrape Website
## Documentation
- Class name: `LLMJinaScrape`
- Category: `SALT/Language Toolkit/Readers`
- Output node: `False`

The LLMJinaScrape node is designed to fetch and process web content from a specified URL using the Jina AI service. It handles HTTP requests, manages potential errors, and formats the response into a structured document, facilitating the integration of web data into further processing pipelines.
## Input types
### Required
- **`url`**
    - The URL to be scraped. This parameter is crucial for defining the target web page from which the content is to be fetched, directly influencing the node's output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`result`**
    - Comfy dtype: `STRING`
    - The raw text content retrieved from the specified URL, serving as a direct representation of the web page's content.
    - Python dtype: `str`
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - A structured document containing the scraped web content along with additional metadata, such as the source and Jina AI URLs, enhancing data context and utility.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMJinaScrape:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {}),
            }
        }
    
    RETURN_TYPES = ("STRING", "DOCUMENT")
    RETURN_NAMES = ("result", "documents")

    FUNCTION = "scrape"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Readers"

    def scrape(self, url):
        jina_url = f"https://r.jina.ai/{url}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response_text = "Empty response"
        try:
            response = requests.get(jina_url, headers=headers)
            response.raise_for_status()
            response_text = response.text
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error occurred: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Connection error occurred: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Timeout error occurred: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"An error occurred: {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {str(e)}")

        document = [Document(text=response_text, extra_info={"jina_url": jina_url, "source_url": url})]

        return (response_text, document)

```
