# ∞ Tavily Search Tool
## Documentation
- Class name: `CreateTavilySearchTool`
- Category: `SALT/Language Toolkit/Agents/Tools`
- Output node: `False`

This node encapsulates the functionality to create a customizable search tool powered by the Tavily search engine. It allows for the dynamic creation of search tools with specific configurations, such as search depth, maximum results, and domain inclusions or exclusions, tailored to enhance search accuracy and relevance for various applications.
## Input types
### Required
- **`api_key`**
    - The API key required to authenticate requests with the Tavily search engine, enabling access to its search capabilities.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`function_name`**
    - The name assigned to the created search tool, which identifies it within the system for future reference and usage.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`search_depth`**
    - Determines the depth of the search performed by the Tavily search engine, affecting how extensively the engine searches through content to find relevant results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`max_results`**
    - The maximum number of search results to return, controlling the breadth of information retrieved from a search query.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`include_answer`**
    - A flag indicating whether to include a direct answer to the search query in the results, providing a concise response alongside traditional search results.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_raw_content`**
    - A flag indicating whether to include the raw content of search results, offering detailed insights into the content matched by the search query.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`include_domains`**
    - A list of domains to specifically include in the search results, focusing the search on preferred sources of information.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_domains`**
    - A list of domains to exclude from the search results, filtering out unwanted or irrelevant sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`tool`**
    - Comfy dtype: `TOOL`
    - The search tool created, encapsulating the configured search functionality and parameters for use in search operations.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateTavilySearchTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "dynamicPrompts": False, "placeholder": "Tavily API key"}),
                "function_name": ("STRING", {"multiline": False, "dynamicPrompts": False, "default": "tavily_search_tool"}),
                "search_depth": (["basic", "advanced"],),
                "max_results": ("INT", {"default": 5}),
                "include_answer": ("BOOLEAN", {"default": False},),
                "include_raw_content": ("BOOLEAN", {"default": False},),
            },
            "optional": {
                "include_domains": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "A list of domains to specifically include in the search results. Default is None, which includes all domains. e.g. \"google.com, twitter.com\"",
                }),
                "exclude_domains": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": False,
                    "placeholder": "A list of domains to specifically exclude from the search results. Default is None, which doesn't exclude any domains. e.g. \"google.com, twitter.com\"",
                }),
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
        search_depth,
        max_results,
        include_answer,
        include_raw_content,
        include_domains=None,
        exclude_domains=None
    ):
        tavily = TavilyClient(api_key=api_key)

        def search(query: str):
            response = tavily.search(
                query=query,
                search_depth=search_depth,
                max_results=max_results,
                include_images=False,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
                include_domains=include_domains.split(
                    ", ") if include_domains is not None and include_domains != "" else None,
                exclude_domains=exclude_domains.split(
                    ", ") if include_domains is not None and exclude_domains != "" else None,
            )
            context = [obj["content"] for obj in response['results']]
            return "/n".join(context)

        tool = {
            "name": function_name,
            "description": "A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current events. Input should be a search query.",
            "function": search,
        }
        return (tool,)

```
