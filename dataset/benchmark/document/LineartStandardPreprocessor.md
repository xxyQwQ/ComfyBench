- `LineartStandardPreprocessor`: The LineartStandardPreprocessor node is designed for extracting line art from images using a standard approach. It preprocesses images to highlight and refine line details, making it suitable for applications requiring clear line delineation, such as digital art and animation.
    - Inputs:
        - `image` (Required): The input image to be processed for line art extraction. Type should be `IMAGE`.
        - `guassian_sigma` (Optional): Specifies the sigma value for the Gaussian blur applied to the image, affecting the smoothness of the lines extracted. Type should be `FLOAT`.
        - `intensity_threshold` (Optional): Determines the threshold for intensity differentiation, influencing the distinction between line art and the background. Type should be `INT`.
        - `resolution` (Optional): The resolution to which the input image is resized before processing, affecting the detail level of the extracted line art. Type should be `INT`.
    - Outputs:
        - `image`: Produces an image with enhanced and refined line art, suitable for further processing or direct use in various applications. Type should be `IMAGE`.