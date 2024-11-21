- `LLMJsonRepair`: The LLMJsonRepair node is designed to correct malformed JSON strings using a language model. It takes a potentially incorrect JSON input and optional directions, and outputs a repaired version of the JSON, ensuring data integrity and proper formatting.
    - Inputs:
        - `llm_model` (Required): Specifies the language model to use for repairing the JSON. It is crucial for interpreting the input and generating the corrected output. Type should be `LLM_MODEL`.
        - `text_input` (Required): The malformed JSON string that needs to be repaired. This input is essential for the node to understand what needs fixing. Type should be `STRING`.
        - `extra_directions` (Optional): Optional additional instructions for the language model to follow during the repair process, allowing for more tailored corrections. Type should be `STRING`.
    - Outputs:
        - `json_output`: The repaired JSON string, corrected for syntax and formatting errors to ensure validity. Type should be `STRING`.