- `AV_LLMApiConfig`: The AV_LLMApiConfig node is designed to generate configuration settings for various large language models (LLMs) by specifying model type, maximum token count, and temperature. This configuration is essential for tailoring the behavior of LLMs to specific tasks or preferences, providing a foundation for further interactions with these models.
    - Inputs:
        - `model` (Required): Specifies the model to be used for the LLM configuration. This parameter supports a wide range of models, including GPT, Claude, and Bedrock variants, allowing for flexible model selection based on the task at hand. Type should be `COMBO[STRING]`.
        - `max_token` (Required): Defines the maximum number of tokens the LLM can generate or process in a single request. This parameter helps control the length of the output, ensuring it meets specific requirements or limitations. Type should be `INT`.
        - `temperature` (Required): Controls the creativity or randomness of the LLM's responses. A higher temperature leads to more varied outputs, while a lower temperature results in more deterministic and predictable text. Type should be `FLOAT`.
    - Outputs:
        - `llm_config`: The output is a configuration object tailored for LLM interactions, encapsulating the specified model, token limit, and temperature settings. Type should be `LLM_CONFIG`.