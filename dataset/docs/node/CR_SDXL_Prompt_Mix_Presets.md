- `CR SDXL Prompt Mix Presets`: This node is designed to blend and manipulate text prompts and styles according to predefined presets, enabling the creation of nuanced and varied input prompts for generative models. It allows for the customization of positive and negative prompts and styles, offering a versatile tool for creative and targeted prompt engineering.
    - Parameters:
        - `prompt_positive`: The positive prompt text to be mixed or manipulated. It serves as the base for generating positive aspects of the final prompt, influencing the generative model's output towards desired characteristics. Type should be `STRING`.
        - `prompt_negative`: The negative prompt text to be mixed or manipulated. It acts as the base for generating negative aspects of the final prompt, guiding the generative model away from undesired characteristics. Type should be `STRING`.
        - `style_positive`: The positive style text to be mixed with the prompt. It enhances the positive aspects of the final prompt by adding stylistic elements, further directing the model's output. Type should be `STRING`.
        - `style_negative`: The negative style text to be mixed with the prompt. It enhances the negative aspects of the final prompt by adding stylistic elements, further preventing undesired outcomes in the model's output. Type should be `STRING`.
        - `preset`: The preset configuration for mixing the prompts and styles. It determines the specific way in which the positive and negative prompts and styles are combined, offering various creative possibilities. Type should be `COMBO[STRING]`.
    - Inputs:
    - Outputs:
        - `pos_g`: The generated positive prompt for global context. Type should be `STRING`.
        - `pos_l`: The generated positive prompt for local context. Type should be `STRING`.
        - `pos_r`: The generated positive prompt for right-aligned context. Type should be `STRING`.
        - `neg_g`: The generated negative prompt for global context. Type should be `STRING`.
        - `neg_l`: The generated negative prompt for local context. Type should be `STRING`.
        - `neg_r`: The generated negative prompt for right-aligned context. Type should be `STRING`.
        - `show_help`: A URL providing additional help and documentation for using the node. Type should be `STRING`.