- `FilterStylerAdvanced`: The SDXLPromptStyler node applies advanced styling to text prompts through a combination of user-selected options, enhancing or modifying the original prompts to achieve desired aesthetic or thematic effects. It supports a dynamic selection of styling options, allowing for a high degree of customization in the styling process.
    - Parameters:
        - `text_positive_g`: The global positive text prompt to be styled, serving as part of the base content for styling transformations. Type should be `STRING`.
        - `text_positive_l`: The local positive text prompt to be styled, working alongside the global prompt to refine the thematic direction of the styled text. Type should be `STRING`.
        - `text_negative`: The negative text prompt to be styled, used to specify content that should be avoided or contrasted against in the styled output. Type should be `STRING`.
        - `filter`: unknown Type should be `COMBO[STRING]`.
        - `negative_prompt_to`: Specifies the scope of the negative styling application, whether it affects global, local, or both types of prompts. Type should be `COMBO[STRING]`.
        - `log_prompt`: When enabled, logs the original and styled prompts along with the user's menu selections, aiding in debugging and understanding the impact of selected styles. Type should be `BOOLEAN`.
    - Inputs:
    - Outputs:
        - `text_positive_g`: The styled version of the global positive text prompt. Type should be `STRING`.
        - `text_positive_l`: The styled version of the local positive text prompt. Type should be `STRING`.
        - `text_positive`: A combined styled version of the global and local positive text prompts. Type should be `STRING`.
        - `text_negative_g`: The styled version of the global negative text prompt, if applicable based on the 'negative_prompt_to' selection. Type should be `STRING`.
        - `text_negative_l`: The styled version of the local negative text prompt, if applicable based on the 'negative_prompt_to' selection. Type should be `STRING`.
        - `text_negative`: The styled version of the negative text prompt, showcasing the effects of the applied filters in contrast or avoidance to the original content. Type should be `STRING`.