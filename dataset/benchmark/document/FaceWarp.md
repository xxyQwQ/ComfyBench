- `FaceWarp`: The FaceWarp node is designed to perform facial warping between two images by utilizing facial landmarks to align features from a source image to a target image. This process involves calculating a transformation matrix based on the landmarks, applying the transformation to warp the source image and its mask to match the target image's geometry, and blending the images for a seamless transition.
    - Inputs:
        - `analysis_models` (Required): A collection of models used for analyzing facial features, including landmark detection and other preprocessing tasks. It plays a crucial role in determining the transformation matrix for warping. Type should be `ANALYSIS_MODELS`.
        - `image_from` (Required): The source image from which facial features will be warped. Type should be `IMAGE`.
        - `image_to` (Required): The target image to which the source image's facial features will be aligned. Type should be `IMAGE`.
        - `keypoints` (Required): Specifies the set of facial landmarks to be used for the warping process, influencing the precision and areas of alignment. Type should be `COMBO[STRING]`.
        - `grow` (Required): A parameter that controls the expansion of the mask around the detected facial landmarks, affecting the area of the image to be warped. Type should be `INT`.
        - `blur` (Required): Determines the level of blurring applied to the edges of the warped image and mask, enhancing the blending effect. Type should be `INT`.
    - Outputs:
        - `image`: The resulting image after applying the warping and blending processes, where the source image's features have been aligned to the target image. Type should be `IMAGE`.
        - `mask`: The mask generated during the warping process, indicating the areas of the source image that have been transformed and blended into the target image. Type should be `MASK`.