- `ImageGaussianBlur`: The `ImageGaussianBlur` node applies a Gaussian blur filter to a collection of images, effectively smoothing them by a specified radius. This operation is commonly used in image processing to reduce noise and detail, or to create a visual effect.
    - Inputs:
        - `images` (Required): The collection of images to be blurred. This input is crucial for defining the set of images that will undergo the Gaussian blur transformation. Type should be `IMAGE`.
        - `radius` (Required): Specifies the radius of the Gaussian blur. A larger radius results in a more pronounced blurring effect. Type should be `INT`.
    - Outputs:
        - `image`: The blurred images, returned as a single tensor by concatenating the individually blurred images along the batch dimension. Type should be `IMAGE`.
