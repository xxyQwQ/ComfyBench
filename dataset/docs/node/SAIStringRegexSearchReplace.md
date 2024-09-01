- `SAIStringRegexSearchReplace`: This node provides functionality for searching and replacing text in a given input string based on a specified regular expression pattern. It allows for the dynamic alteration of text content by identifying patterns and substituting them with a desired replacement string.
    - Parameters:
        - `text_input`: The input text where the search and replacement operation will be performed. It serves as the primary content for regex operations. Type should be `STRING`.
        - `regex_pattern`: The regular expression pattern used to identify the text segments within the input text that need to be replaced. Type should be `STRING`.
        - `replacement_text`: The text that will replace the segments identified by the regex pattern in the input text. Type should be `STRING`.
    - Inputs:
    - Outputs:
        - `replaced_text`: The resulting text after the search and replace operations have been performed on the input text. Type should be `STRING`.