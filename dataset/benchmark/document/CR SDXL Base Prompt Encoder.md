- `CR SDXL Base Prompt Encoder`: The CR SDXL Base Prompt Encoder node is designed to encode base prompts for the SDXL model, providing foundational text inputs that can be further customized or mixed with other prompts. It plays a crucial role in establishing the initial context or theme for text-to-image generation tasks within the SDXL framework.
    - Inputs:
        - `base_clip` (Required): Represents the foundational CLIP model used for encoding the base prompts, setting the initial visual-textual context for the SDXL model. Type should be `CLIP`.
        - `pos_g` (Required): Positive guidance text for enhancing specific attributes or themes in the generated images. Type should be `STRING`.
        - `pos_l` (Required): Localized positive guidance text for fine-tuning specific areas or aspects within the generated images. Type should be `STRING`.
        - `neg_g` (Required): Negative guidance text for suppressing undesired attributes or themes in the generated images. Type should be `STRING`.
        - `neg_l` (Required): Localized negative guidance text for reducing specific undesired areas or aspects within the generated images. Type should be `STRING`.
        - `preset` (Required): A predefined set of parameters or settings that influence the encoding process and the resulting image generation. Type should be `COMBO[STRING]`.
        - `base_width` (Required): The width of the base image or context area for the encoding process. Type should be `INT`.
        - `base_height` (Required): The height of the base image or context area for the encoding process. Type should be `INT`.
        - `crop_w` (Required): Width of the cropping area applied to the base context, allowing for focus on specific regions during encoding. Type should be `INT`.
        - `crop_h` (Required): Height of the cropping area applied to the base context, enabling emphasis on particular regions during encoding. Type should be `INT`.
        - `target_width` (Required): The target width for the output image, guiding the scaling and aspect ratio of the generated content. Type should be `INT`.
        - `target_height` (Required): The target height for the output image, directing the scaling and aspect ratio of the generated content. Type should be `INT`.
    - Outputs:
        - `base_positive`: The encoded positive base prompt, ready for further processing or combination with other prompts for image generation. Type should be `CONDITIONING`.
        - `base_negative`: The encoded negative base prompt, prepared for additional processing or merging with other prompts for image generation. Type should be `CONDITIONING`.
        - `show_help`: A helper output providing guidance or suggestions for optimizing the use of encoded prompts in the SDXL model. Type should be `STRING`.