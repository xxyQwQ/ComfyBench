- `Text Parse Noodle Soup Prompts`: This node is designed to parse and transform text inputs using Noodle Soup Prompts (NSP) or wildcard patterns. It dynamically replaces specified placeholders or wildcards in the input text with random selections from a predefined set of terms or patterns, allowing for the generation of varied and contextually relevant text outputs.
    - Inputs:
        - `mode` (Required): Specifies the parsing mode: either 'Noodle Soup Prompts' for NSP-based parsing or 'Wildcards' for wildcard pattern replacement. This choice determines the method of text transformation. Type should be `COMBO[STRING]`.
        - `noodle_key` (Required): The delimiter used to identify NSP terms or wildcard patterns within the input text. It serves as a marker for the start and end of placeholders to be replaced. Type should be `STRING`.
        - `seed` (Required): A seed value for the random number generator, ensuring reproducibility of the text transformation process by controlling the randomness of selections. Type should be `INT`.
        - `text` (Required): The input text to be parsed and transformed according to the specified mode and patterns. Type should be `STRING`.
    - Outputs:
        - `string`: The transformed text output, with NSP terms or wildcard patterns replaced according to the specified mode and selections. Type should be `STRING`.
