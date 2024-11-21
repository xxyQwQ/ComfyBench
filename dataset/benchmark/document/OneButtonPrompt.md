- `OneButtonPrompt`: The OneButtonPrompt node streamlines the creation and customization of prompts for image generation, offering a suite of options to tailor the prompt's subject, style, and complexity. It leverages a dynamic prompt building mechanism to generate prompts that can vary widely in theme and detail, accommodating a broad range of creative needs.
    - Inputs:
        - `insanitylevel` (Required): Determines the level of creativity and randomness in the generated prompt, influencing its complexity and uniqueness. Type should be `INT`.
        - `artist` (Optional): Selects artists to influence the style and aesthetic of the generated image, contributing to the prompt's thematic depth. Type should be `COMBO[STRING]`.
        - `imagetype` (Optional): Defines the type of image to be generated, such as digital art, painting, or concept art, guiding the visual style of the output. Type should be `COMBO[STRING]`.
        - `imagemodechance` (Optional): Controls the likelihood of selecting a special image mode for the prompt, adding an element of randomness to the image style. Type should be `INT`.
        - `subject` (Optional): Specifies the main subject for the prompt, offering a base around which the prompt is constructed. Type should be `COMBO[STRING]`.
        - `custom_subject` (Optional): Allows for the specification of a custom subject around which the prompt will be built, providing a focused thematic direction. Type should be `STRING`.
        - `custom_outfit` (Optional): Specifies an outfit to be included in the prompt, adding detail to the character description. Type should be `STRING`.
        - `prompt_prefix` (Optional): Specifies a prefix to be added to the beginning of the prompt, allowing for further customization. Type should be `STRING`.
        - `prompt_suffix` (Optional): Specifies a suffix to be added to the end of the prompt, enabling additional tailoring of the prompt's theme. Type should be `STRING`.
        - `humanoids_gender` (Optional): Determines the gender of humanoid subjects in the prompt, allowing for more specific character depiction. Type should be `COMBO[STRING]`.
        - `emojis` (Optional): Enables or disables the inclusion of emojis in the prompt, affecting its tone and readability. Type should be `COMBO[BOOLEAN]`.
        - `base_model` (Optional): Chooses the base model for prompt generation, affecting the style and language of the prompt. Type should be `COMBO[STRING]`.
        - `prompt_enhancer` (Optional): Selects a prompt enhancer to modify the prompt's style or theme, offering more nuanced control over the generated output. Type should be `COMBO[STRING]`.
        - `seed` (Optional): Sets a seed for the random number generator, ensuring reproducibility of prompts when the same inputs are provided. Type should be `INT`.
    - Outputs:
        - `prompt`: The primary generated prompt, ready for use in creative applications. Type should be `STRING`.
        - `prompt_g`: A variant of the generated prompt, offering an alternative perspective or theme. Type should be `STRING`.
        - `prompt_l`: Another variant of the generated prompt, providing additional creative options. Type should be `STRING`.