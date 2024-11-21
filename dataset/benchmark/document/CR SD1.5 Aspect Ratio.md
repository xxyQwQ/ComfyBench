- `CR SD1.5 Aspect Ratio`: This node is designed to adjust the aspect ratio of images for Stable Diffusion 1.5, allowing users to select from a variety of predefined aspect ratios, swap dimensions, and manage upscale factors. It's particularly useful for tailoring image dimensions to specific requirements or preferences.
    - Inputs:
        - `width` (Required): The initial width of the image. This value is adjusted based on the selected aspect ratio. Type should be `INT`.
        - `height` (Required): The initial height of the image. This value is adjusted based on the selected aspect ratio. Type should be `INT`.
        - `aspect_ratio` (Required): A predefined aspect ratio selection that determines the new dimensions of the image. Type should be `COMBO[STRING]`.
        - `swap_dimensions` (Required): A toggle to swap the width and height of the image, allowing for easy orientation changes. Type should be `COMBO[STRING]`.
        - `upscale_factor` (Required): A factor by which the image is upscaled, affecting the final image size. Type should be `FLOAT`.
        - `batch_size` (Required): The number of images to process in a batch, affecting performance and memory usage. Type should be `INT`.
    - Outputs:
        - `width`: The adjusted width of the image after applying the selected aspect ratio. Type should be `INT`.
        - `height`: The adjusted height of the image after applying the selected aspect ratio. Type should be `INT`.
        - `upscale_factor`: The factor by which the image has been upscaled. Type should be `FLOAT`.
        - `batch_size`: The number of images processed in the batch. Type should be `INT`.
        - `empty_latent`: The latent representation of the image with adjusted dimensions, ready for further processing or generation. Type should be `LATENT`.
        - `show_help`: A URL to the help documentation for this node. Type should be `STRING`.