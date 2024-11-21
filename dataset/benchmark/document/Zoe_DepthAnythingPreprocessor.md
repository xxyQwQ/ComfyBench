- `Zoe_DepthAnythingPreprocessor`: This node preprocesses images for depth estimation by selecting and applying a depth estimation model based on the specified environment (indoor or outdoor). It leverages the ZoeDepthAnythingDetector model to generate depth maps, enhancing the understanding of spatial relationships in images.
    - Inputs:
        - `image` (Required): The input image to be processed for depth estimation. Type should be `IMAGE`.
        - `environment` (Optional): Determines the choice of pretrained model for depth estimation, selecting between indoor and outdoor environments to optimize depth map accuracy. Type should be `COMBO[STRING]`.
        - `resolution` (Optional): Specifies the resolution at which the depth estimation should be performed. Affects the output depth map's resolution. Type should be `INT`.
    - Outputs:
        - `image`: The output is a depth map of the input image, providing a pixel-wise estimation of depth values. Type should be `IMAGE`.