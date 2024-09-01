- `SaltMaskDilateRegion`: This node applies a dilation filter to mask regions, effectively expanding the areas of interest within the masks based on the specified number of iterations. It's designed to process and modify mask regions to highlight or enlarge specific features within the masks.
    - Parameters:
        - `iterations`: Specifies the number of times the dilation operation is applied to the masks. This parameter controls the extent of dilation, affecting the size and visibility of features within the masks. Type should be `INT`.
    - Inputs:
        - `masks`: The input masks to be dilated. This parameter is crucial for defining the areas within the image that will undergo dilation, directly impacting the node's output. Type should be `MASK`.
    - Outputs:
        - `MASKS`: The output masks after dilation. These masks represent the modified regions with expanded areas of interest, showcasing the effect of the dilation process. Type should be `MASK`.