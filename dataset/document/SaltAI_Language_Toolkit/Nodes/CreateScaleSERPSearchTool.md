# ∞ Scale SERP Research Tool
## Documentation
- Class name: `CreateScaleSERPSearchTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node facilitates the creation of a search tool powered by ScaleSERP, designed to perform search engine queries optimized for comprehensive, accurate, and trusted results. It is particularly useful for answering questions related to current events, leveraging a variety of parameters such as search type, location, and device preferences to tailor the search results.
## Input types
### Required
- **`api_key`**
    - The API key required to authenticate requests with the ScaleSERP service. It is crucial for enabling the search functionality of the tool.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`function_name`**
    - The name assigned to the created search tool. It serves as an identifier and is used to reference the tool within the system.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`search_type`**
    - Specifies the type of search to be performed, influencing the nature of the results returned. It allows for customization of the search to fit specific needs or contexts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`location`**
    - An optional parameter that specifies the geographical location for the search, enabling localized search results.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`device`**
    - Defines the type of device the search is optimized for, such as desktop, mobile, or tablet, affecting the format and relevance of the search results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mobile_type`**
    - Specifies the type of mobile device for the search when the device is set to mobile, further customizing the search experience.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tablet_type`**
    - Specifies the type of tablet device for the search when the device is set to tablet, further customizing the search experience.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`tool`**
    - Comfy dtype: `TOOL`
    - The search tool created, encapsulating the functionality to perform searches using ScaleSERP with the specified parameters.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateScaleSERPSearchTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "SALTAI_SERP_KEY"}),
                "function_name": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "scale_serp_search_tool"}),
                "search_type": (["none", "news", "videos", "scholar", "places", "shopping"],),
            },
            "optional": {
                "location": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Location for search results (e.g., 'United States')"}),
                "device": (["desktop", "tablet", "mobile"],),
                "mobile_type": (["iphone", "android"],),
                "tablet_type": (["ipad", "android"],),
            }
        }

    RETURN_TYPES = ("TOOL",)
    RETURN_NAMES = ("tool",)

    FUNCTION = "create_tool"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Agents/Tools"

    def create_tool(
        self,
        api_key,
        function_name,
        search_type,
        location=None,
        device="desktop",
        mobile_type="iphone",
        tablet_type="ipad",
    ):
        client = ScaleSERP(api_key=api_key)

        def search(query: str):
            kwargs = {
                "search_type": search_type if search_type != "none" else None,
                "output": "json",
                "gl": location,
                "device": device,
                "mobile_type": mobile_type,
                "tablet_type": tablet_type,
            }

            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            response = client.search(query=query, hide_base64_images=True, **kwargs)

            logger.info("SERP Response")
            logger.data(response)

            if 'results' in response and isinstance(response['results'], list):
                context = [obj["title"] for obj in response['results']]
                return "\n".join(context)
            else:
                return f"No results found or unexpected response format:\n{str(response)}"

        tool = {
            "name": function_name,
            "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
            "function": search,
        }
        return (tool,)

```
