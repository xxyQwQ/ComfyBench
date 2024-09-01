- `Inference_Core_LineartStandardPreprocessor`: The Inference_Core_LineartStandardPreprocessor node is designed to preprocess images for line extraction, applying Gaussian blurring and intensity thresholding to enhance lineart features before further processing.
    - Parameters:
        - `guassian_sigma`: unknown Type should be `FLOAT`.
        - `intensity_threshold`: Determines the threshold for intensity differentiation, aiding in the distinction of lineart from the background by setting a cutoff intensity value. Type should be `INT`.
        - `resolution`: Specifies the resolution at which the image processing should be executed, affecting the detail level of the output lineart. Type should be `INT`.
    - Inputs:
        - `image`: The input image to be processed for line extraction, serving as the primary data for the node's operations. Type should be `IMAGE`.
    - Outputs:
        - `image`: Produces an image with enhanced lineart features, ready for further processing or analysis. Type should be `IMAGE`.