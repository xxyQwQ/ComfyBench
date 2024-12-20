- `Text2InputOr3rdOption`: This node processes three input texts and, based on a condition, outputs either two of the original texts or duplicates one across both outputs. It allows for dynamic text manipulation and conditional output generation.
    - Inputs:
        - `text_a` (Required): The first text input to be potentially modified and outputted. It serves as one of the primary inputs for conditional processing. Type should be `STRING`.
        - `text_b` (Required): The second text input that may be modified and outputted, acting as another primary input for the node's conditional logic. Type should be `STRING`.
        - `text_c` (Required): The third text input, which can replace the other two inputs based on the condition specified by 'use_text_c_for_both'. Type should be `STRING`.
        - `use_text_c_for_both` (Required): A boolean flag determining whether 'text_c' should be used as the output for both 'text_a' and 'text_b', enabling conditional output behavior. Type should be `COMBO[STRING]`.
    - Outputs:
        - `text_a`: The modified version of 'text_a', or 'text_c' if the condition is met. Type should be `STRING`.
        - `text_b`: The modified version of 'text_b', or 'text_c' if the condition is met. Type should be `STRING`.
