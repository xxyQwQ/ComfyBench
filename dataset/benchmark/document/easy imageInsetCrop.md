- `easy imageInsetCrop`: The node provides functionality for cropping an image based on specified inset dimensions, adjusting the crop area according to the given measurements and ensuring the dimensions do not exceed the image's boundaries. It supports cropping by percentage or absolute values, with additional adjustments to align the crop area to a grid, enhancing compatibility with certain image processing operations.
    - Inputs:
        - `image` (Required): The image to be cropped, provided as a multi-dimensional array representing the image data. Type should be `IMAGE`.
        - `measurement` (Required): Specifies the unit of measurement for the crop dimensions, allowing for either percentage-based or absolute value cropping, which influences how the crop boundaries are calculated. Type should be `COMBO[STRING]`.
        - `left` (Required): The left boundary of the crop area, defined either as a percentage of the image's width or as an absolute value, depending on the measurement unit. Type should be `INT`.
        - `right` (Required): The right boundary of the crop area, similar to 'left', but for the right side of the image. Type should be `INT`.
        - `top` (Required): The top boundary of the crop area, defined in the same manner as 'left' and 'right', but for the top edge of the image. Type should be `INT`.
        - `bottom` (Required): The bottom boundary of the crop area, defined similarly to the other boundaries, for the bottom edge of the image. Type should be `INT`.
    - Outputs:
        - `image`: The cropped image, returned as a multi-dimensional array with the same structure as the input image but with dimensions adjusted according to the specified crop boundaries. Type should be `IMAGE`.