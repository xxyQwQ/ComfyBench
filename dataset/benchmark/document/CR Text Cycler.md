- `CR Text Cycler`: The CR Text Cycler node is designed to repeat and loop through given text lines, creating a list of text items based on specified repetition and loop counts. It facilitates the generation of repetitive text sequences for various applications, such as animations or list manipulations.
    - Inputs:
        - `text` (Required): The 'text' parameter takes a multiline string input, which is split into lines to be cycled through. It serves as the base content for repetition and looping, forming the core of the text sequences generated by the node. Type should be `STRING`.
        - `repeats` (Required): The 'repeats' parameter specifies how many times each line of text is to be repeated within a single loop. It controls the density of repetition for each text item, affecting the overall length and composition of the output list. Type should be `INT`.
        - `loops` (Required): The 'loops' parameter determines the number of times the entire set of text lines (after applying the 'repeats' parameter) is looped over. It influences the final size of the output list by repeating the sequence of text items multiple times. Type should be `INT`.
    - Outputs:
        - `STRING`: This output is a list of text items generated by cycling through the input text lines according to the specified 'repeats' and 'loops' parameters. It represents the repeated and looped text sequence. Type should be `*`.
        - `show_text`: A URL providing additional help and documentation for using the CR Text Cycler node. Type should be `STRING`.