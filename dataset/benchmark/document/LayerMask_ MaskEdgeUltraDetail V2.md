- `LayerMask_ MaskEdgeUltraDetail V2`: This node specializes in enhancing the detail and precision of mask edges in layer masks, utilizing advanced techniques to refine edges for improved visual clarity and accuracy. It offers a sophisticated approach to mask edge processing, catering to needs for high-detail and precision in image editing tasks.
    - Inputs:
        - `image` (Required): The input image(s) to be processed. This parameter is crucial for determining the areas where mask edge refinement is applied, directly influencing the outcome of the node's operation. Type should be `IMAGE`.
        - `mask` (Required): The mask(s) corresponding to the input image(s), indicating areas of interest for edge refinement. The precision of mask edges is enhanced based on this parameter, making it essential for targeted detail improvement. Type should be `MASK`.
        - `method` (Required): Specifies the technique used for mask edge refinement, affecting the level of detail and precision achieved in the processed mask edges. Type should be `COMBO[STRING]`.
        - `mask_grow` (Required): Determines the extent to which mask edges are expanded, playing a key role in the overall edge refinement process. Type should be `INT`.
        - `fix_gap` (Required): Controls the gap filling in mask edges, aiding in creating smoother and more continuous edges. Type should be `INT`.
        - `fix_threshold` (Required): Sets the threshold for fixing gaps in mask edges, contributing to the refinement of edge detail and continuity. Type should be `FLOAT`.
        - `edge_erode` (Required): Defines the amount of erosion applied to mask edges, crucial for achieving the desired level of edge detail and precision. Type should be `INT`.
        - `edte_dilate` (Required): Specifies the dilation applied to mask edges after erosion, balancing the edge refinement process by restoring some of the eroded details. Type should be `INT`.
        - `black_point` (Required): Adjusts the black point in the image, affecting the contrast and visibility of mask edges. Type should be `FLOAT`.
        - `white_point` (Required): Adjusts the white point in the image, influencing the brightness and clarity of mask edges. Type should be `FLOAT`.
        - `device` (Required): Indicates the computing device (CPU or GPU) used for processing, affecting the performance and efficiency of the node. Type should be `COMBO[STRING]`.
        - `max_megapixels` (Required): Limits the maximum size of images processed, ensuring efficient memory usage and performance. Type should be `FLOAT`.
    - Outputs:
        - `image`: The processed image(s) with refined mask edges, showcasing enhanced detail and precision. Type should be `IMAGE`.
        - `mask`: The refined mask(s) with improved edge detail, corresponding to the processed image(s). Type should be `MASK`.