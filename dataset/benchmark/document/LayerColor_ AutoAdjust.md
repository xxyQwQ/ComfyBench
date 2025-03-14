- `LayerColor_ AutoAdjust`: The AutoAdjust node provides automated adjustments to the color properties of an image, including brightness, contrast, saturation, and individual color channels. It aims to enhance the image's overall appearance or correct its color balance based on specified parameters.
    - Inputs:
        - `image` (Required): The input image to be adjusted. This parameter is crucial as it serves as the base for all subsequent color corrections and enhancements. Type should be `IMAGE`.
        - `strength` (Required): Determines the intensity of the automatic adjustments applied to the image, affecting the overall impact of the correction. Type should be `INT`.
        - `brightness` (Required): Adjusts the brightness level of the image, influencing its lightness or darkness. Type should be `INT`.
        - `contrast` (Required): Modifies the contrast of the image, enhancing or reducing the difference between its light and dark areas. Type should be `INT`.
        - `saturation` (Required): Alters the saturation level, adjusting the intensity of the image's colors. Type should be `INT`.
        - `red` (Required): Adjusts the red color channel, allowing for fine-tuning of the image's red hues. Type should be `INT`.
        - `green` (Required): Adjusts the green color channel, enabling modifications to the image's green tones. Type should be `INT`.
        - `blue` (Required): Adjusts the blue color channel, facilitating adjustments to the blue hues in the image. Type should be `INT`.
    - Outputs:
        - `image`: The output image after applying the auto adjustments, reflecting changes in color properties such as brightness, contrast, saturation, and color channels. Type should be `IMAGE`.
