- `CreateTavilySearchTool`: This node encapsulates the functionality to create a customizable search tool powered by the Tavily search engine. It allows for the dynamic creation of search tools with specific configurations, such as search depth, maximum results, and domain inclusions or exclusions, tailored to enhance search accuracy and relevance for various applications.
    - Inputs:
        - `api_key` (Required): The API key required to authenticate requests with the Tavily search engine, enabling access to its search capabilities. Type should be `STRING`.
        - `function_name` (Required): The name assigned to the created search tool, which identifies it within the system for future reference and usage. Type should be `STRING`.
        - `search_depth` (Required): Determines the depth of the search performed by the Tavily search engine, affecting how extensively the engine searches through content to find relevant results. Type should be `COMBO[STRING]`.
        - `max_results` (Required): The maximum number of search results to return, controlling the breadth of information retrieved from a search query. Type should be `INT`.
        - `include_answer` (Required): A flag indicating whether to include a direct answer to the search query in the results, providing a concise response alongside traditional search results. Type should be `BOOLEAN`.
        - `include_raw_content` (Required): A flag indicating whether to include the raw content of search results, offering detailed insights into the content matched by the search query. Type should be `BOOLEAN`.
        - `include_domains` (Optional): A list of domains to specifically include in the search results, focusing the search on preferred sources of information. Type should be `STRING`.
        - `exclude_domains` (Optional): A list of domains to exclude from the search results, filtering out unwanted or irrelevant sources. Type should be `STRING`.
    - Outputs:
        - `tool`: The search tool created, encapsulating the configured search functionality and parameters for use in search operations. Type should be `TOOL`.