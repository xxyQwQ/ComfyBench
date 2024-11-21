- `ColorCorrect`: The ColorCorrect node is designed to adjust and enhance the color properties of an image, including temperature, hue, brightness, contrast, saturation, and gamma. It utilizes advanced image processing techniques to fine-tune these attributes, improving the overall visual quality of the image.
    - Inputs:
        - `image` (Required): The input image tensor to be color corrected. It serves as the base for applying various color adjustments. Type should be `IMAGE`.
        - `temperature` (Required): Adjusts the color temperature of the image, making it warmer or cooler. Type should be `FLOAT`.
        - `hue` (Required): Modifies the hue of the image, shifting the colors along the color spectrum. Type should be `FLOAT`.
        - `brightness` (Required): Controls the brightness level of the image, making it lighter or darker. Type should be `FLOAT`.
        - `contrast` (Required): Alters the contrast of the image, enhancing the difference between light and dark areas. Type should be `FLOAT`.
        - `saturation` (Required): Adjusts the saturation level, affecting the intensity of the colors in the image. Type should be `FLOAT`.
        - `gamma` (Required): Applies gamma correction to the image, adjusting the luminance or brightness. Type should be `FLOAT`.
    - Outputs:
        - `image`: The color-corrected image tensor, reflecting the applied adjustments. Type should be `IMAGE`.