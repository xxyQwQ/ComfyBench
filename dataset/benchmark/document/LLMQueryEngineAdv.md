- `LLMQueryEngineAdv`: The LLMQueryEngineAdv node is designed to leverage language models for advanced querying capabilities. It constructs a comprehensive query from user inputs and messages, utilizes an embedding model for query expansion, and employs a vector index retriever with post-processing for similarity-based filtering. This node aims to provide precise and relevant responses by integrating language understanding and retrieval technologies.
    - Inputs:
        - `llm_model` (Required): The language model and optional embedding model used for query expansion and understanding. It's crucial for interpreting the query and retrieving relevant information. Type should be `LLM_MODEL`.
        - `llm_index` (Required): The index used for retrieving query results based on vector similarity. Essential for the node's ability to find the most relevant responses. Type should be `LLM_INDEX`.
        - `query` (Optional): The user's query, which is combined with other messages to form the complete query input. It plays a key role in directing the search and retrieval process. Type should be `STRING`.
        - `llm_message` (Optional): A list of messages that can be included in the query construction. These messages add context and detail to the query, enhancing its comprehensiveness. Type should be `LIST`.
        - `top_k` (Optional): Specifies the number of top similar results to retrieve. It determines the breadth of the search results. Type should be `INT`.
        - `similarity_cutoff` (Optional): The similarity threshold for filtering results. Only results with a similarity score above this cutoff are considered relevant. Type should be `FLOAT`.
    - Outputs:
        - `results`: The retrieved and processed query response, which is expected to be relevant and precise based on the input query and similarity criteria. Type should be `STRING`.
