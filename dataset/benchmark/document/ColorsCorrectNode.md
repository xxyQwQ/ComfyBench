- `ColorsCorrectNode`: The ColorsCorrectNode is designed for image color correction, allowing adjustments to brightness, contrast, saturation, gamma, and hue. It supports optional color tinting using a specified hex color value, providing a comprehensive suite of image enhancement tools.
    - Inputs:
        - `image` (Required): The input image to be corrected. It serves as the base for all subsequent color and enhancement operations. Type should be `IMAGE`.
        - `brightness` (Required): Adjusts the brightness of the image. A higher value increases brightness. Type should be `FLOAT`.
        - `contrast` (Required): Adjusts the contrast of the image. A higher value increases contrast. Type should be `FLOAT`.
        - `saturation` (Required): Adjusts the saturation of the image, enhancing or muting the colors. Type should be `FLOAT`.
        - `gamma` (Required): Adjusts the gamma of the image, affecting the luminance of mid-tones. Type should be `FLOAT`.
        - `hue_degrees` (Required): Adjusts the hue of the image in degrees. This shifts all colors in the image. Type should be `FLOAT`.
        - `use_color` (Required): A boolean flag that determines whether to apply a color tint to the image based on the specified hex color. Type should be `BOOLEAN`.
        - `hex_color` (Optional): The hex color code used for tinting the image if use_color is True. It allows for custom color adjustments. Type should be `STRING`.
    - Outputs:
        - `image`: The corrected image after applying the specified adjustments. Type should be `IMAGE`.
