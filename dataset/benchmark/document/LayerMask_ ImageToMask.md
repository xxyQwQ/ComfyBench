- `LayerMask_ ImageToMask`: The `LayerMask: ImageToMask` node is designed to transform images into masks, focusing on extracting and processing mask layers from given images. It utilizes image processing techniques to create or modify mask layers, which can be used for further image manipulation or analysis.
    - Inputs:
        - `image` (Required): The `image` parameter specifies the input image from which the mask will be generated. It is the primary source for mask creation. Type should be `IMAGE`.
        - `channel` (Required): The `channel` parameter selects the specific color channel or image property to be used for mask generation, such as luminance or specific color channels, influencing the mask's characteristics. Type should be `COMBO[STRING]`.
        - `black_point` (Required): The `black_point` parameter sets the lower threshold for mask generation, converting pixels darker than this value to black in the resulting mask. Type should be `INT`.
        - `white_point` (Required): The `white_point` parameter sets the upper threshold for mask generation, converting pixels lighter than this value to white in the resulting mask. Type should be `INT`.
        - `gray_point` (Required): The `gray_point` parameter adjusts the mid-tone levels in the mask, affecting the contrast and detail of the mask. Type should be `FLOAT`.
        - `invert_output_mask` (Required): The `invert_output_mask` parameter allows for the inversion of the generated mask, swapping its black and white areas. Type should be `BOOLEAN`.
        - `mask` (Optional): The `mask` parameter, when provided, specifies an existing mask to be modified or combined with the newly generated mask from the image. Type should be `MASK`.
    - Outputs:
        - `mask`: The output `mask` is the transformed or newly created mask layer, which has been processed according to the specified parameters. It can be used for further image manipulation or analysis. Type should be `MASK`.