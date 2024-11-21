- `BatchNormalizeImage`: The BatchNormalizeImage node is designed to normalize a batch of images based on a given factor, adjusting each image's pixel values to have a standard deviation and mean that aligns more closely with the batch's overall characteristics. This process enhances the consistency of image data, making it more suitable for further processing or analysis.
    - Inputs:
        - `images` (Required): The 'images' parameter represents the batch of images to be normalized. It is crucial for the normalization process as it directly influences the computation of mean and standard deviation used for normalization. Type should be `IMAGE`.
        - `factor` (Required): The 'factor' parameter controls the extent to which the original images are blended with their normalized versions. It plays a significant role in determining the final appearance of the normalized images. Type should be `FLOAT`.
    - Outputs:
        - `image`: The output is a batch of images that have been normalized according to the specified factor, potentially enhancing their suitability for further image processing tasks. Type should be `IMAGE`.