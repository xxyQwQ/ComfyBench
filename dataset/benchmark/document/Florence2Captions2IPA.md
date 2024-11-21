- `Florence2Captions2IPA`: The Florence2Captions2IPA node is designed to format input captions into a standardized string format. It accepts captions in various forms, including single strings or lists, and processes them into a unified string representation. This functionality is essential for preparing text data for further processing or analysis, ensuring consistency in input format.
    - Inputs:
        - `caption_input` (Required): The 'caption_input' parameter can accept captions in multiple formats, including single strings or lists of strings. It plays a crucial role in determining how the input text is formatted for further processing, directly impacting the node's output. Type should be `STRING`.
    - Outputs:
        - `string`: The output is a formatted string, which may be a concatenation of multiple input strings or a single input string, depending on the input type. This standardized output is suitable for subsequent text processing or analysis tasks. Type should be `STRING`.