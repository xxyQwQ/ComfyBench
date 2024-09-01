- `DepthStylerAdvanced`: The SDXLPromptStyler node is designed for advanced styling of text prompts, enabling customization through a variety of styling options. It allows for the modification of both positive and negative aspects of prompts, supporting detailed adjustments to enhance the overall impact and effectiveness of the generated content.
    - Parameters:
        - `text_positive_g`: Represents the global positive aspects of the text prompt, focusing on broad, overarching positive themes or elements to be emphasized in the generated content. Type should be `STRING`.
        - `text_positive_l`: Captures the local positive aspects of the text prompt, detailing specific positive elements or details to be highlighted. Type should be `STRING`.
        - `text_negative`: This input captures the negative aspects or elements the user wants to minimize or avoid in the generated content, ensuring the styling adjustments counteract these undesired features. Type should be `STRING`.
        - `depth`: unknown Type should be `COMBO[STRING]`.
        - `negative_prompt_to`: Specifies the scope of the negative prompt adjustments, allowing users to target either global, local, or both aspects for negative styling. Type should be `COMBO[STRING]`.
        - `log_prompt`: A boolean flag that, when enabled, logs the original and styled prompts along with the user's selections for each menu, providing insight into how the styling choices influence the final output. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `text_positive_g`: The enhanced global positive text prompt, styled to emphasize overarching positive themes more effectively. Type should be `STRING`.
        - `text_positive_l`: The enhanced local positive text prompt, styled to highlight specific positive details more effectively. Type should be `STRING`.
        - `text_positive`: The combined global and local positive aspects of the text prompt, styled to emphasize both broad themes and specific details more effectively. Type should be `STRING`.
        - `text_negative_g`: The adjusted global negative text prompt, modified to further suppress or negate undesired overarching negative elements. Type should be `STRING`.
        - `text_negative_l`: The adjusted local negative text prompt, modified to further suppress or negate undesired specific negative details. Type should be `STRING`.
        - `text_negative`: The combined global and local negative aspects of the text prompt, adjusted to suppress or negate both broad themes and specific details more effectively. Type should be `STRING`.