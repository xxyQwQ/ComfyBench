- `GradientToFloat`: Transforms an image into a list of float values based on sampling along its width and height axes. This process is useful for extracting numerical representations of images for further analysis or processing.
    - Inputs:
        - `image` (Required): The image tensor to be sampled, where the sampling occurs along the width and height axes to generate float values. Type should be `IMAGE`.
        - `steps` (Required): Defines the number of intervals to sample along the width and height axes of the image, affecting the resolution of the output float lists. Type should be `INT`.
    - Outputs:
        - `float_x`: A list of float values representing the mean of the sampled points along the width of the image. Type should be `FLOAT`.
        - `float_y`: A list of float values representing the mean of the sampled points along the height of the image. Type should be `FLOAT`.