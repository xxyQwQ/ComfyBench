- `CR Mask Text`: This node is designed to apply text masking operations within a graphical context, enabling the creation of text-based masks that can be used for various image manipulation and composition tasks. It abstracts the complexities involved in text rendering and mask generation, providing a straightforward way to integrate text into visual designs programmatically.
    - Inputs:
        - `image` (Required): The input image over which the text mask will be applied. This forms the base layer for the text masking operation. Type should be `IMAGE`.
        - `text` (Required): The text content to be used for creating the mask. This parameter allows users to specify the exact text that will be rendered into the mask. Type should be `STRING`.
        - `font_name` (Required): Specifies the font to be used for rendering the text. This allows for customization of the text appearance in the mask. Type should be `COMBO[STRING]`.
        - `font_size` (Required): Determines the size of the font used for the text. This affects how large the text appears within the mask. Type should be `INT`.
        - `background_color` (Required): The color of the background over which the text is rendered. This parameter is crucial for defining the visual contrast between the text and the background. Type should be `COMBO[STRING]`.
        - `align` (Required): Specifies the alignment of the text within the mask. This can be left, center, or right, affecting the text's positioning relative to the specified margins. Type should be `COMBO[STRING]`.
        - `justify` (Required): Determines whether the text is justified within the mask. This affects the distribution of space between words in each line. Type should be `COMBO[STRING]`.
        - `margins` (Required): Specifies the margins around the text within the mask. This parameter helps in positioning the text properly within the mask. Type should be `INT`.
        - `line_spacing` (Required): Controls the spacing between lines of text. This is important for adjusting the readability of the text within the mask. Type should be `INT`.
        - `position_x` (Required): The horizontal position of the text within the mask. This parameter allows for precise placement of the text. Type should be `INT`.
        - `position_y` (Required): The vertical position of the text within the mask. Similar to position_x, it enables accurate placement of the text. Type should be `INT`.
        - `rotation_angle` (Required): The angle at which the text is rotated. This parameter allows for dynamic text orientations within the mask. Type should be `FLOAT`.
        - `rotation_options` (Required): Additional options for controlling the rotation of the text. This provides further customization of the text's appearance. Type should be `COMBO[STRING]`.
        - `bg_color_hex` (Optional): Hexadecimal representation of the background color. Provides an alternative way to specify the background color. Type should be `STRING`.
    - Outputs:
        - `IMAGE`: The resulting image after the text masking operation, showcasing the applied text mask over the original image. Type should be `IMAGE`.
        - `show_help`: unknown Type should be `STRING`.