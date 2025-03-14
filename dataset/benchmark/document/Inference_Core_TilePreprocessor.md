- `Inference_Core_TilePreprocessor`: The Tile Preprocessor node is designed to enhance image inputs for further processing by applying a tiling mechanism. This involves detecting and adjusting image tiles to improve the quality and consistency of the input images for subsequent stages in a pipeline, particularly in control networks.
    - Inputs:
        - `image` (Required): The input image to be processed and enhanced through the tiling mechanism. It serves as the primary data upon which the tile detection and adjustment operations are performed. Type should be `IMAGE`.
        - `pyrUp_iters` (Optional): Specifies the number of iterations for the pyramid upscaling process, affecting the granularity of the tile adjustment. This parameter plays a crucial role in determining the level of detail and the scale of adjustments applied to the input image. Type should be `INT`.
        - `resolution` (Optional): The target resolution for the output image, influencing the final size and detail level after processing. It determines how the image is resized as part of the preprocessing steps. Type should be `INT`.
    - Outputs:
        - `image`: Produces an enhanced version of the input image, where tiling adjustments have been applied to improve its suitability for further processing steps. Type should be `IMAGE`.
