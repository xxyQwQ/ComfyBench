- `CR Half Drop Panel`: The CR Half Drop Panel node is designed to transform an input image into a patterned panel by applying half-drop, quarter-drop, or custom percentage drop techniques. This node enhances visual aesthetics by creating repetitive patterns that can be used in various design contexts.
    - Inputs:
        - `image` (Required): The input image to be transformed into a patterned panel. This image serves as the base for creating the repetitive pattern. Type should be `IMAGE`.
        - `pattern` (Required): Specifies the type of pattern to apply to the input image, such as 'half drop', 'quarter drop', or a 'custom drop %', determining the arrangement of the repeated image segments. Type should be `COMBO[STRING]`.
        - `drop_percentage` (Optional): Used when 'pattern' is set to 'custom drop %', it defines the percentage by which the image is dropped, allowing for customizable pattern repetition. Type should be `FLOAT`.
    - Outputs:
        - `image`: The output image after applying the specified drop pattern, showcasing the transformed panel with the repetitive design. Type should be `IMAGE`.
        - `show_help`: A URL to the documentation or help page for the CR Half Drop Panel node, providing additional information and guidance. Type should be `STRING`.
