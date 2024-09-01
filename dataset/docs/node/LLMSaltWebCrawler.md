- `LLMSaltWebCrawler`: The LLMSaltWebCrawler node is designed for web crawling and scraping, capable of navigating through web pages to collect data. It supports features like depth control, domain exclusion, SSL verification, and keyword-based relevancy filtering to efficiently gather and parse content from specified URLs.
    - Parameters:
        - `url`: The starting point URL for the web crawl. It's crucial for defining the scope of the crawl and serves as the entry point for the crawling process. Type should be `STRING`.
        - `max_depth`: Defines how deep the crawler should navigate from the starting URL. It limits the crawl to a specified depth to manage the scope and resources. Type should be `INT`.
        - `max_links`: Limits the number of links to follow per page, controlling the volume of data collected and managing resource usage. Type should be `INT`.
        - `trim_line_breaks`: A flag to indicate whether to remove line breaks from the text extracted during the crawl, aiding in data cleanliness. Type should be `BOOLEAN`.
        - `verify_ssl`: Determines whether to verify SSL certificates when making requests, enhancing security by avoiding potentially harmful sites. Type should be `BOOLEAN`.
        - `exclude_domains`: Specifies domains to exclude from the crawl, allowing for more targeted data collection by avoiding irrelevant or unwanted sites. Type should be `STRING`.
        - `keywords`: A list of keywords used to filter content based on relevancy, ensuring the data collected is pertinent to specific interests or topics. Type should be `STRING`.
    - Inputs:
        - `urls`: A list of URLs to be crawled. This allows for multiple entry points, expanding the breadth of the crawl. Type should be `LIST`.
    - Outputs:
        - `documents`: The output is a list of documents, each containing structured data from the crawled web pages, including URLs, titles, texts, and links found, organized for easy processing and analysis. Type should be `DOCUMENT`.