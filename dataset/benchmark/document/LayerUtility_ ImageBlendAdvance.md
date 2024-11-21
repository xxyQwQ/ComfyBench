- `LayerUtility_ ImageBlendAdvance`: The ImageBlendAdvance node is designed for advanced blending of images within a layer-based editing context. It leverages sophisticated algorithms to merge images seamlessly, offering users enhanced control over the blending process for creating complex visual compositions.
    - Inputs:
        - `background_image` (Required): The base image over which the layer image will be blended. It serves as the canvas for the blending operation. Type should be `IMAGE`.
        - `layer_image` (Required): The image to be blended onto the background image. This layer can be manipulated through various parameters to achieve the desired blending effect. Type should be `IMAGE`.
        - `invert_mask` (Required): A boolean flag that, when set, inverts the blending mask, affecting how the layer image merges with the background. Type should be `BOOLEAN`.
        - `blend_mode` (Required): Defines the algorithm used for blending the layer image with the background, influencing the visual outcome of the blend. Type should be `COMBO[STRING]`.
        - `opacity` (Required): Determines the transparency level of the layer image, allowing for finer control over its visibility against the background. Type should be `INT`.
        - `x_percent` (Required): Specifies the horizontal positioning of the layer image relative to the background, enabling precise alignment. Type should be `FLOAT`.
        - `y_percent` (Required): Specifies the vertical positioning of the layer image relative to the background, enabling precise alignment. Type should be `FLOAT`.
        - `mirror` (Required): Allows for the mirroring of the layer image either horizontally or vertically, adding to the creative possibilities. Type should be `COMBO[STRING]`.
        - `scale` (Required): Controls the size of the layer image, enabling scaling up or down as needed for the composition. Type should be `FLOAT`.
        - `aspect_ratio` (Required): Adjusts the aspect ratio of the layer image, maintaining its proportions while scaling. Type should be `FLOAT`.
        - `rotate` (Required): Rotates the layer image around its center, offering additional compositional flexibility. Type should be `FLOAT`.
        - `transform_method` (Required): Specifies the interpolation method used during transformations such as scaling or rotating, affecting image quality. Type should be `COMBO[STRING]`.
        - `anti_aliasing` (Required): Sets the level of anti-aliasing applied to the layer image, enhancing the visual quality of edges and transitions. Type should be `INT`.
        - `layer_mask` (Optional): unknown Type should be `MASK`.
    - Outputs:
        - `image`: The final blended image resulting from the application of the specified parameters and blending operations. Type should be `IMAGE`.
        - `mask`: An optional output that provides the mask used in the blending process, useful for further editing or analysis. Type should be `MASK`.