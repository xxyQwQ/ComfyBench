- `SaltSplitByNewline`: The node splits a given text into a list of strings based on newline characters, optionally stripping whitespace and ignoring lines that start with common comment symbols (#, //).
    - Inputs:
        - `text` (Required): The text to be split into lines. This input allows for the processing of multiline text, enabling the node to operate on extensive or compact text data. Type should be `STRING`.
        - `strip_text` (Required): A boolean flag that determines whether whitespace should be stripped from the beginning and end of each line, as well as ignoring lines that start with comment symbols. Type should be `BOOLEAN`.
    - Outputs:
        - `list`: A list of strings derived from the input text, split based on newline characters and optionally processed according to the strip_text parameter. Type should be `LIST`.
        - `strings`: A duplicate of the list output, provided for compatibility or further processing needs. Type should be `STRING`.
