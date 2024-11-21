- `LayerUtility_ ImageScaleByAspectRatio`: This node is designed to adjust the scale of an image based on its aspect ratio, offering various scaling options to fit, fill, or match specific dimensions while maintaining the original or a custom aspect ratio. It supports different resizing methods to optimize the visual quality of the scaled image.
    - Inputs:
        - `aspect_ratio` (Required): Specifies the aspect ratio to use for scaling the image. It can be set to 'original' to use the image's current aspect ratio, 'custom' to use a specified ratio, or a predefined ratio in the format 'width:height'. Type should be `COMBO[STRING]`.
        - `proportional_width` (Required): The width part of the custom aspect ratio, used when 'aspect_ratio' is set to 'custom'. Type should be `INT`.
        - `proportional_height` (Required): The height part of the custom aspect ratio, used when 'aspect_ratio' is set to 'custom'. Type should be `INT`.
        - `fit` (Required): Determines how the image should be fitted to the target dimensions, affecting how the image is scaled and cropped. Type should be `COMBO[STRING]`.
        - `method` (Required): The method used for resizing the image, such as 'bicubic', 'hamming', 'bilinear', 'box', or 'nearest', each offering different quality and performance characteristics. Type should be `COMBO[STRING]`.
        - `round_to_multiple` (Required): If specified, rounds the scaled dimensions to the nearest multiple of this value, which can be useful for aligning images to certain size constraints. Type should be `COMBO[STRING]`.
        - `scale_to_longest_side` (Required): Determines if the image should be scaled based on the longest side, providing a boolean option to enable or disable this feature. Type should be `BOOLEAN`.
        - `longest_side` (Required): Specifies the target length for the longest side of the image when 'scale_to_longest_side' is enabled, determining the final dimensions of the scaled image. Type should be `INT`.
        - `image` (Optional): The original image to be scaled. This can be a tensor representation of an image. Type should be `IMAGE`.
        - `mask` (Optional): An optional mask to be scaled alongside the image, typically used for segmentation tasks. Type should be `MASK`.
    - Outputs:
        - `image`: The scaled image tensor, adjusted according to the specified aspect ratio, dimensions, and scaling method. Type should be `IMAGE`.
        - `mask`: The scaled mask tensor, corresponding to the 'mask' input, adjusted to match the dimensions of the scaled image. Type should be `MASK`.
        - `original_size`: The original dimensions of the image before scaling. Type should be `BOX`.
        - `width`: The width of the scaled image. Type should be `INT`.
        - `height`: The height of the scaled image. Type should be `INT`.