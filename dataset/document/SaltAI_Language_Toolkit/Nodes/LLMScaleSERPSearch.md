# ∞ Scale SERP Search
## Documentation
- Class name: `LLMScaleSERPSearch`
- Category: `SALT/Language Toolkit/Tools`
- Output node: `False`

This node is designed to leverage the capabilities of the ScaleSERP module to perform search operations, parsing the results into a structured format suitable for further analysis or integration into larger workflows. It abstracts the complexities of interacting with the ScaleSERP API, providing a streamlined interface for executing searches and handling the results.
## Input types
### Required
- **`api_key`**
    - The api_key is a required string input that represents the API key for accessing the ScaleSERP service. It is essential for authenticating and authorizing the search operation on the platform.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`query`**
    - The query is a required string input that specifies the search terms or phrases. It is the primary determinant of the search operation's scope and relevance, directly influencing the results retrieved from the platform.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`search_type`**
    - The search_type input allows users to specify the type of search to perform, such as 'news', 'scholar', 'places', or 'shopping', affecting the scope and nature of the search results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`location`**
    - The location input enables specifying the geographical location for the search, influencing the relevance of the search results to the specified area.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`device`**
    - The device input specifies the type of device context for the search, such as 'desktop', 'mobile', or 'tablet', affecting the format and relevance of the search results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mobile_type`**
    - The mobile_type input is relevant when the device is set to 'mobile', allowing further specification of the mobile device type for tailored search results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tablet_type`**
    - The tablet_type input is relevant when the device is set to 'tablet', allowing further specification of the tablet device type for tailored search results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - The documents output contains the structured search results, including relevant information parsed from the ScaleSERP platform, suitable for further analysis or integration.
    - Python dtype: `list[dict]`
- **`results_dict`**
    - Comfy dtype: `DICT`
    - The results_dict output contains the raw JSON response from the ScaleSERP API, offering comprehensive access to all data retrieved during the search operation.
    - Python dtype: `dict`
- **`links_list`**
    - Comfy dtype: `LIST`
    - The links_list output includes the URLs of the search results, providing direct links to the sources of information retrieved during the search operation.
    - Python dtype: `list[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMScaleSERPSearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {}),
                "query": ("STRING", {}),
            },
            "optional": {
                "search_type": (["none", "news", "videos", "scholar", "places", "shopping"],),
                "location": ("STRING", {}),
                "device": (["desktop", "tablet", "mobile"],),
                "mobile_type": (["iphone", "android"],),
                "tablet_type": (["ipad", "android"],),
            }
        }

    RETURN_TYPES = ("DOCUMENT", "DICT", "LIST")
    RETURN_NAMES = ("documents", "results_dict", "links_list")

    FUNCTION = "search"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def search(self, api_key, query, **kwargs):

        if kwargs.__contains__("search_type"):
            if kwargs["search_type"] == "none":
                kwargs.pop("search_type")

        if kwargs.__contains__("device"):
            if kwargs["device"] == "desktop" and kwargs.__contains__("mobile_type") and kwargs.__contains__("tablet_type"):
                kwargs.pop("mobile_type")
                kwargs.pop("tablet_type")
            if kwargs["device"] == "mobile":
                if kwargs.__contains__("tablet_type"):
                    kwargs.pop("tablet_type")
            if kwargs["device"] == "tablet":
                if kwargs.__contains__("mobile_type"):
                    kwargs.pop("mobile_type")

        client = ScaleSERP(api_key=api_key)
        results = client.search(query, hide_base64_images=True, **kwargs)

        documents = client.results_to_documents()
        for doc in documents:
            logger.info(f"Text:\n{doc.text}, Metadata:\n{doc.extra_info}\n==================\n")
        links = client.results_to_link_dict()

        return (documents, results, list(links.values()))

```
