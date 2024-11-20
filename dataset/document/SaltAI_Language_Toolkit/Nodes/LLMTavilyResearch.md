---
tags:
- LLM
---

# ∞ Tavily Research
## Documentation
- Class name: `LLMTavilyResearch`
- Category: `SALT/Language Toolkit/Tools`
- Output node: `False`

This node is designed to leverage the capabilities of the Tavily Research model, focusing on advanced language understanding and generation tasks. It aims to provide deep insights and analyses, potentially enhancing various applications with sophisticated language processing features.
## Input types
### Required
- **`tavily_api_key`**
    - Specifies the API key required to access the Tavily Research model, essential for authenticating and enabling the node's operations.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`search_query`**
    - The query input for conducting searches, serving as the basis for retrieving relevant information and insights.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`search_depth`**
    - Determines the depth of the search, affecting the comprehensiveness of the results obtained.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`max_results`**
    - Limits the number of results returned, allowing for control over the volume of data processed.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`include_answer`**
    - A flag to include direct answers within the search results, enhancing the utility of the output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_raw_content`**
    - A flag to include the raw content of the search results, providing access to unprocessed data.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`include_domains`**
    - Specifies domains to be included in the search results, focusing the search on specific sources.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`exclude_domains`**
    - Specifies domains to be excluded from the search results, refining the scope of the search.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`keep_looking_limit`**
    - Sets a limit on the number of times the search will attempt to find more results beyond the initial set, affecting the thoroughness of the search.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - A collection of documents retrieved as a result of the search, providing structured insights.
    - Python dtype: `list`
- **`urls`**
    - Comfy dtype: `LIST`
    - A list of URLs corresponding to the documents found, offering direct links to the source material.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LLMTavilyResearch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tavily_api_key": ("STRING", {"default": "tvly-*******************************"}),
                "search_query": ("STRING", {"multiline": True, "dynamicPrompts": False}),
            },
            "optional": {
                "search_depth": (["basic", "advanced"],),
                "max_results": ("INT", {"min": 1, "max": 20, "default": 1}),
                "include_answer": ("BOOLEAN", {"default": False},),
                "include_raw_content": ("BOOLEAN", {"default": False},),
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
                "keep_looking_limit": ("INT", {"min": 1, "max": 20, "default": 10})
            }
        }
    
    RETURN_TYPES = ("DOCUMENT", "LIST")
    RETURN_NAMES = ("documents", "urls")

    FUNCTION = "search"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def search(self, tavily_api_key, search_query, search_depth="basic", max_results=1, include_answer=False, include_raw_content=False, include_domains="google.com", exclude_domains=None, keep_looking_limit=10):
        
        tavily = TavilyClient(api_key=tavily_api_key)

        def tavily_search():
            return tavily.search(
                query=search_query,
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
        
        logger.info("Tavily Search Query:", search_query)

        # Increment the search results because when using `include_raw_content` 
        # results are found in order of accessibility, so first X results may not 
        # be traversible, and end up in there being no result to return. But maybe 
        # the next search result does have traversible content.
        adjusted_max_results = max_results + keep_looking_limit
        current_retry = 0
        response = {}
        while "results" not in response or not response["results"] and max_results < adjusted_max_results:
                max_results += 1
                if current_retry > 0:
                    logger.warning(f"Unable find any results. Continuing Search...\nRetry {current_retry} of {keep_looking_limit}")
                response = tavily_search()
                current_retry += 1

        results = response.get("results", None)
        urls = []
        documents = []
        if results:
            for result in results:
                content = result.pop("content", "null")
                raw_content = result.pop("raw_content", None)
                document = Document(
                    text=(raw_content if raw_content else content),
                    extra_info=result
                )
                documents.append(document)
                if result.__contains__("url"):
                    urls.append(result["url"])
        else:
            documents.append(Document(text="No document data available", extra_info={"error": "No document data available"}))
            urls.append(None)

        return (documents, urls)

```
