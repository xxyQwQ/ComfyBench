- `Mask Dilate Region`: This node is designed to dilate regions within given masks, effectively expanding the areas of interest. It utilizes iterations to control the extent of dilation, allowing for precise adjustments to the mask's features.
    - Parameters:
        - `iterations`: Specifies the number of times the dilation operation is applied, directly influencing the degree of expansion for the mask's regions. Type should be `INT`.
    - Inputs:
        - `masks`: The input masks to be dilated. This parameter is crucial for defining the areas within the image that will undergo dilation. Type should be `MASK`.
    - Outputs:
        - `MASKS`: The output consists of dilated masks, where the specified regions have been expanded according to the number of iterations. Type should be `MASK`.