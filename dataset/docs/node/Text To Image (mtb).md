- `Text To Image (mtb)`: The Text To Image node is designed to convert text into images using specified fonts. It searches for font files within a specified directory structure, supporting a variety of font formats, and allows for customization of the text appearance in the generated image. This node is useful for dynamically creating images from text for various applications, such as generating labels, captions, or visual representations of textual data.
    - Parameters:
        - `text`: The text string to be converted into an image. This parameter is crucial as it defines the content of the resulting image. Type should be `STRING`.
        - `font`: Specifies the font to be used for the text in the image. This parameter allows for customization of the text's appearance. Type should be `COMBO[STRING]`.
        - `wrap`: Determines whether the text should wrap to fit within the specified width of the image. Type should be `BOOLEAN`.
        - `trim`: Indicates whether to trim whitespace around the text in the generated image. Type should be `BOOLEAN`.
        - `line_height`: Adjusts the line height of the text, affecting how closely lines of text are spaced. Type should be `FLOAT`.
        - `font_size`: Specifies the size of the font used for the text, directly influencing the text's appearance in the image. Type should be `INT`.
        - `width`: The width of the generated image in pixels. Type should be `INT`.
        - `height`: The height of the generated image in pixels. Type should be `INT`.
        - `h_offset`: Horizontal offset of the text from its aligned position, allowing for fine-tuned positioning. Type should be `INT`.
        - `v_offset`: Vertical offset of the text from its aligned position, enabling precise control over the text's placement. Type should be `INT`.
        - `h_coverage`: Percentage of the image width that the text is allowed to cover, affecting text wrapping and layout. Type should be `INT`.
    - Inputs:
        - `color`: Defines the color of the text in the image. Type should be `COLOR`.
        - `background`: Specifies the background color of the image. Type should be `COLOR`.
        - `h_align`: Horizontal alignment of the text within the image. Type should be `['left', 'center', 'right']`.
        - `v_align`: Vertical alignment of the text within the image. Type should be `['top', 'center', 'bottom']`.
    - Outputs:
        - `image`: The output is an image generated from the provided text, incorporating the specified font, layout, and styling options. Type should be `IMAGE`.