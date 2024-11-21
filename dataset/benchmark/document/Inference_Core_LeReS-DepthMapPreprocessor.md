- `Inference_Core_LeReS-DepthMapPreprocessor`: The Inference_Core_LeReS-DepthMapPreprocessor node is designed for preprocessing images to generate depth maps using the LeReS algorithm. It enhances image depth perception by optionally boosting the depth estimation process and applying specific adjustments to remove nearest objects or background, aiming to improve the depth quality for further processing or visualization.
    - Inputs:
        - `image` (Required): The input image to be processed for depth map generation. Type should be `IMAGE`.
        - `rm_nearest` (Optional): Specifies the threshold for removing the nearest objects in the depth map, enhancing the focus on more distant elements. Type should be `FLOAT`.
        - `rm_background` (Optional): Defines the threshold for background removal in the depth map, helping to isolate the main subjects from their surroundings. Type should be `FLOAT`.
        - `boost` (Optional): Enables or disables the boost mode for depth estimation, where 'enable' significantly enhances the depth map details. Type should be `COMBO[STRING]`.
        - `resolution` (Optional): The resolution at which the depth map should be generated, affecting the detail level of the output. Type should be `INT`.
    - Outputs:
        - `image`: The processed depth map image, optimized for further computational tasks or visualization. Type should be `IMAGE`.