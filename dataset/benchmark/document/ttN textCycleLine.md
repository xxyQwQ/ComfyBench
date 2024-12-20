- `ttN textCycleLine`: The ttN textCycleLine node is designed for cycling through lines of text based on a given index and control method. It allows for dynamic text manipulation by selecting specific lines from a larger text body, supporting operations like increment, decrement, random selection, or fixed position access.
    - Inputs:
        - `text` (Required): The 'text' parameter is the input text from which lines will be cycled through. It plays a crucial role in determining the content available for cycling and directly impacts the output based on the selected line. Type should be `STRING`.
        - `index` (Required): The 'index' parameter specifies the starting point or specific line to access within the input text. Its value influences which line is selected during the cycling process. Type should be `INT`.
        - `index_control` (Required): The 'index_control' parameter determines the method of cycling through lines, such as incrementing, decrementing, randomizing, or fixing the index. This affects how the next line is selected from the input text. Type should be `COMBO[STRING]`.
    - Outputs:
        - `string`: unknown Type should be `STRING`.
