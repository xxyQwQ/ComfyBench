- `Resolution Multiply (JPS)`: The Resolution Multiply node is designed to scale the dimensions of an image by a specified factor, effectively multiplying the width and height of the image by this factor to produce new dimensions. This node is useful in scenarios where image resolution adjustments are required, such as preparing images for processing or display at different sizes.
    - Inputs:
        - `width` (Required): Specifies the original width of the image. The width is scaled by the factor to calculate the new width. Type should be `INT`.
        - `height` (Required): Specifies the original height of the image. The height is scaled by the factor to calculate the new height. Type should be `INT`.
        - `factor` (Required): The multiplier used to scale the width and height of the image. A factor greater than 1 increases the image size, while a factor less than 1 would decrease it (though the node's constraints do not allow for reduction). Type should be `INT`.
    - Outputs:
        - `width_resized`: The new width of the image after scaling by the specified factor. Type should be `INT`.
        - `height_resized`: The new height of the image after scaling by the specified factor. Type should be `INT`.