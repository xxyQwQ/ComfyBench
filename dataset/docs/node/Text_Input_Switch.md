- `Text Input Switch`: The Text Input Switch node is designed to selectively output one of two text inputs based on a boolean condition. It serves as a logical switch within text processing workflows, enabling conditional text flow and decision-making.
    - Parameters:
        - `text_a`: The first text input option. This input, along with 'text_b', forms the pair between which the node chooses based on the boolean condition. Type should be `STRING`.
        - `text_b`: The second text input option. Together with 'text_a', it provides the alternative text input for the node to select from, contingent on the boolean condition. Type should be `STRING`.
        - `boolean`: A boolean input that determines which text input ('text_a' or 'text_b') is passed through as the output. True will pass 'text_a', and False will pass 'text_b'. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `string`: Outputs the selected text input based on the boolean condition, effectively allowing conditional text routing within a workflow. Type should be `STRING`.