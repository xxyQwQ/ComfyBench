# ∞ Web Crawler
## Documentation
- Class name: `LLMSaltWebCrawler`
- Category: `SALT/Language Toolkit/Tools`
- Output node: `False`

The LLMSaltWebCrawler node is designed for web crawling and content extraction, tailored to efficiently navigate and retrieve information from websites. It leverages advanced parsing techniques to handle different content types (HTML, XML, JSON), assesses page relevance based on specified keywords, and manages link exploration depth and breadth with customizable parameters. This node is adept at extracting structured data from web pages, making it a valuable tool for web scraping, data mining, and content analysis tasks.
## Input types
### Required
### Optional
- **`url`**
    - The URL to start the crawl from. It serves as the entry point for the web crawling process, determining the initial web page from which the crawler begins its operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`urls`**
    - A list of URLs to be crawled. This allows for multiple entry points for the crawling process, enabling broader coverage of web content.
    - Comfy dtype: `LIST`
    - Python dtype: `list`
- **`max_depth`**
    - Specifies the maximum depth of the crawl, controlling how deep the crawler can go into website links from the starting point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`max_links`**
    - The maximum number of links to follow from a single page. This parameter helps in controlling the breadth of the crawl, ensuring a focused and efficient web scraping process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`trim_line_breaks`**
    - Indicates whether to remove line breaks from the extracted content, which can help in cleaning and standardizing the text data.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`verify_ssl`**
    - Determines whether to verify the SSL certificates of the websites being crawled. This can ensure the security of the web crawling process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`exclude_domains`**
    - A list of domains to exclude from the crawl. This helps in avoiding unwanted or irrelevant content by preventing the crawler from accessing specified domains.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`keywords`**
    - Keywords to filter the content. Pages containing these keywords are considered relevant, allowing for targeted crawling based on content relevance.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`relevant_links`**
    - Indicates whether to evaluate links for relevance based on the specified keywords. This helps in refining the crawl to include only links that are likely to contain relevant information.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`relevant_page_content`**
    - Determines whether to evaluate page content for relevance. This allows for the exclusion of pages that do not contain any of the specified keywords, focusing the crawl on relevant content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_threads`**
    - The maximum number of threads to use for parallel crawling. This parameter enhances the efficiency of the crawling process by enabling concurrent fetching of web content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`use_jina`**
    - A flag indicating whether to use Jina for scraping. This enables the use of Jina's scraping capabilities, potentially improving the efficiency and effectiveness of content retrieval.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`documents`**
    - Comfy dtype: `DOCUMENT`
    - A collection of documents representing the extracted web page details, including URLs, titles, texts, and links found on the pages. It encapsulates the data mined during the crawl, structured for easy access and analysis.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LLMSaltWebCrawler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "url": ("STRING", {}),
                "urls": ("LIST", {}),
                "max_depth": ("INT", {"min": 1, "max": 12, "default": 2}),
                "max_links": ("INT", {"min": 1, "max": 100, "default": 2}),
                "trim_line_breaks": ("BOOLEAN", {"default": True}),
                "verify_ssl": ("BOOLEAN", {"default": True}),
                "exclude_domains": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Optional exclude domains, eg: example.com, example2.com\nUse an asterisk (*) to exclude all domains that aren't listed in the input URLs"}),
                "keywords": ("STRING", {"multiline": True, "dynamicPrompts": False, "placeholder": "Optional relevancy keywords, eg: artificial intelligence, ai"}),
                "relevant_links": ("BOOLEAN", {"default": True}),
                "relevant_page_content": ("BOOLEAN", {"default": True}),
                "max_threads": ("INT", {"min": 1, "max": 64, "default": 2}),
                "use_jina": ("BOOLEAN", {"default": False})
            }
        }
    

    RETURN_TYPES = ("DOCUMENT",)
    RETURN_NAMES = ("documents",)

    FUNCTION = "crawl"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Tools"

    def crawl(
            self, 
            url:str="google.com", 
            urls:list=None, 
            max_depth:int=2, 
            max_links:int=2, 
            trim_line_breaks:bool=True, 
            verify_ssl:bool=True, 
            exclude_domains:str="", 
            keywords:str=None, 
            relevant_links:bool=True, 
            relevant_page_content:bool=True, 
            max_threads:int=2,
            use_jina:bool=False
        ) -> list:

        search_urls = []
        if not url.strip() and not urls:
            raise Exception("Please provide at lease one URL")
        
        logger.info(f"Keyword: {list(keywords)}")
        if keywords in ("", "undefined") or keywords is None:
            keywords = None
            relevant_links = False
            relevant_page_content = False
        
        url = url.strip()
        if url != "" and valid_url(url):
            search_urls.append(url)
        if urls:
            search_urls.extend([url for url in urls if valid_url(url)])

        logger.info("Valid URLs:")
        logger.info(search_urls)

        crawler = WebCrawler(
            search_urls, 
            exclude_domains=exclude_domains, 
            keywords=keywords, 
            max_links=max_links, 
            evaluate_links=relevant_links, 
            evaluate_page_content=relevant_page_content, 
            max_threads=max_threads,
            jina_scrape=use_jina
        )

        results = crawler.parse_sites(crawl=True, max_depth=max_depth, trim_line_breaks=trim_line_breaks, verify_ssl=verify_ssl)

        return (results.to_documents(), )

```
