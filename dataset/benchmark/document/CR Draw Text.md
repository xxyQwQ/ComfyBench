- `CR Draw Text`: This node is designed for drawing text onto images or graphics, allowing for customization of font properties, alignment, and positioning. It provides a versatile tool for adding textual elements to visual content, enhancing the ability to convey messages or labels within a graphical context.
    - Inputs:
        - `image_width` (Required): Specifies the width of the image onto which the text will be drawn. This parameter sets the canvas size for the textual content. Type should be `INT`.
        - `image_height` (Required): Specifies the height of the image, defining the vertical dimension of the canvas for the text. Type should be `INT`.
        - `text` (Required): The text to be drawn onto the image. It allows for customization of the content and message conveyed through the graphical output. Type should be `STRING`.
        - `font_name` (Required): Allows for the selection of the font used to render the text, impacting the style and readability. Type should be `COMBO[STRING]`.
        - `font_size` (Required): Determines the size of the text, affecting visibility and emphasis within the image. Type should be `INT`.
        - `font_color` (Required): Specifies the color of the text, crucial for visibility against the background and overall visual appeal. Type should be `COMBO[STRING]`.
        - `background_color` (Required): Defines the background color of the image, setting the visual context for the text. Type should be `COMBO[STRING]`.
        - `align` (Required): Determines the alignment of the text within the image, affecting its positioning and overall layout. Type should be `COMBO[STRING]`.
        - `justify` (Required): Determines the justification of the text, affecting its alignment relative to the specified position. Type should be `COMBO[STRING]`.
        - `margins` (Required): Sets the margins around the text, aiding in its precise positioning within the image. Type should be `INT`.
        - `line_spacing` (Required): Controls the spacing between lines of text, affecting readability and text layout. Type should be `INT`.
        - `position_x` (Required): Specifies the horizontal position of the text within the image, allowing for accurate placement. Type should be `INT`.
        - `position_y` (Required): Specifies the vertical position of the text within the image, enabling precise alignment. Type should be `INT`.
        - `rotation_angle` (Required): Determines the angle at which the text is rotated, enabling dynamic text orientation. Type should be `FLOAT`.
        - `rotation_options` (Required): Provides options for how the text rotation is applied, offering flexibility in text presentation. Type should be `COMBO[STRING]`.
        - `font_color_hex` (Optional): An optional hexadecimal value for the text color, offering an alternative to predefined color selections. Type should be `STRING`.
        - `bg_color_hex` (Optional): An optional hexadecimal value for the background color, providing additional customization options. Type should be `STRING`.
    - Outputs:
        - `IMAGE`: The modified image after the text has been drawn onto it, showcasing the integration of textual and visual elements. Type should be `IMAGE`.
        - `show_help`: Provides a link to additional documentation and help resources related to the node's functionality. Type should be `STRING`.