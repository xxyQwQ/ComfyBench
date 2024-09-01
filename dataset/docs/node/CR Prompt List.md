- `CR Prompt List`: The CR Prompt List node is designed to aggregate and manage a collection of prompts for generative tasks, facilitating the organization and sequential processing of textual inputs for creative or analytical applications.
    - Parameters:
        - `prepend_text`: This parameter allows for the addition of text before the main content of each prompt, enabling customization and contextualization of the input sequence. Type should be `STRING`.
        - `multiline_text`: Accepts multiple lines of text as input, providing a flexible space for entering extensive or detailed prompts that require more than a single line. Type should be `STRING`.
        - `append_text`: Enables the appending of text to the end of each prompt, allowing for further customization or the addition of closing remarks or signatures. Type should be `STRING`.
        - `start_index`: Determines the starting index for processing the list of prompts, offering control over the sequence's beginning point. Type should be `INT`.
        - `max_rows`: Limits the number of prompts to be processed, facilitating the management of large collections by setting a maximum threshold. Type should be `INT`.
    - Inputs:
    - Outputs:
        - `prompt`: Represents the aggregated and potentially transformed collection of prompts, ready for downstream generative tasks. Type should be `STRING`.
        - `body_text`: Provides the main content of the processed prompts, central to the node's output. Type should be `STRING`.
        - `show_help`: Offers a link to additional information or guidance related to the node's functionality, supporting user understanding and application. Type should be `STRING`.