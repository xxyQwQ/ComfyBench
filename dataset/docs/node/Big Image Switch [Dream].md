- `Big Image Switch [Dream]`: This node is designed to switch between different image inputs based on a selection criterion, allowing for dynamic image selection within a workflow.
    - Parameters:
        - `select`: Determines which image input to select based on the provided criterion, enabling dynamic image selection. Type should be `INT`.
        - `on_missing`: Specifies the action to take when the selected image input is missing, ensuring robustness in image selection. Type should be `COMBO[STRING]`.
    - Inputs:
        - `input_i`: Represents one of the multiple image inputs that can be selected. The index 'i' varies, indicating each distinct image input available for selection. Type should be `IMAGE`.
    - Outputs:
        - `selected`: The image that has been selected based on the provided criterion. Type should be `IMAGE`.