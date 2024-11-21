- `SDXLAspectRatioSelector`: The SDXLAspectRatioSelector node is designed to select and adjust the aspect ratio for images, ensuring they fit specific dimensions while maintaining the original proportions. It provides a mechanism to scale images to a wide range of predefined aspect ratios, making it suitable for various display or processing requirements.
    - Inputs:
        - `aspect_ratio` (Required): Specifies the desired aspect ratio for the image, chosen from a predefined list of ratios. This selection determines the dimensions to which the image will be scaled, affecting its final appearance. Type should be `COMBO[STRING]`.
    - Outputs:
        - `ratio`: The selected aspect ratio as a string, indicating the proportion between width and height of the image. Type should be `STRING`.
        - `width`: The calculated width of the image after scaling to the selected aspect ratio. Type should be `INT`.
        - `height`: The calculated height of the image after scaling to the selected aspect ratio. Type should be `INT`.