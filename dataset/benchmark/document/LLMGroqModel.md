- `LLMGroqModel`: The LLMGroqModel node integrates Groq's large language models (LLMs) with embedding models, such as those provided by OpenAI, to facilitate advanced text processing and analysis tasks. It enables the loading of specific LLMs and embedding models, configuring them for use in various applications that require understanding, generating, or transforming text.
    - Inputs:
        - `model` (Required): Specifies the Groq model to be loaded for text processing tasks, indicating the specific LLM to be utilized. Type should be `COMBO[STRING]`.
        - `groq_api_key` (Required): The API key required to authenticate and access Groq's large language models, ensuring secure and authorized use. Type should be `STRING`.
        - `embedding_model` (Required): Defines the embedding model to be used in conjunction with the Groq LLM, typically for tasks involving text embedding or similarity analysis. Type should be `COMBO[STRING]`.
        - `openai_api_key` (Optional): Optional API key for accessing OpenAI's embedding models, providing flexibility in choosing the embedding model based on availability or preference. Type should be `STRING`.
    - Outputs:
        - `llm_model`: The loaded Groq large language model, ready for text processing tasks. Type should be `LLM_MODEL`.
        - `embed_model_only`: The embedding model loaded alongside the Groq LLM, used for text embedding or similarity analysis. Type should be `LLM_EMBED_MODEL`.