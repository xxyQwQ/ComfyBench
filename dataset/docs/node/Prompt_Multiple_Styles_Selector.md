- `Prompt Multiple Styles Selector`: This node is designed to select and concatenate multiple style prompts from a predefined set, allowing for the generation of complex and nuanced text prompts based on user-selected styles.
    - Parameters:
        - `style1`: The first style to be included in the prompt generation. Its selection influences the overall theme and direction of the generated prompt. Type should be `COMBO[STRING]`.
        - `style2`: The second style to be included, further refining and adding to the theme initiated by the first style. Type should be `COMBO[STRING]`.
        - `style3`: The third style selection, adding another layer of nuance to the prompt generation. Type should be `COMBO[STRING]`.
        - `style4`: The fourth and final style choice, completing the composition of the prompt with its unique characteristics. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `positive_string`: The concatenated positive prompt generated from the selected styles. Type should be `STRING`.
        - `negative_string`: The concatenated negative prompt generated from the selected styles, used for refining the generation process. Type should be `STRING`.