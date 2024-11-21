- `Mask Crop Region`: The node focuses on cropping a specified region from a given mask, based on the region type (dominant or minority) and an optional padding. It aims to extract and highlight specific areas within masks, enhancing the focus on regions of interest.
    - Inputs:
        - `mask` (Required): The mask input represents the image mask from which a specific region is to be cropped. It plays a crucial role in determining the area of interest for cropping. Type should be `MASK`.
        - `padding` (Required): Padding specifies the additional space to be added around the cropped region. It allows for more flexibility in the size of the output region, potentially including more context around the targeted area. Type should be `INT`.
        - `region_type` (Required): The region type determines whether the dominant or minority region within the mask is to be cropped. This choice directs the cropping process towards areas of major or minor presence, respectively. Type should be `COMBO[STRING]`.
    - Outputs:
        - `cropped_mask`: The cropped mask is the result of the cropping operation, containing the specified region of interest from the original mask. Type should be `MASK`.
        - `crop_data`: Crop data provides detailed information about the dimensions and coordinates of the cropped region, facilitating further processing or analysis. Type should be `CROP_DATA`.
        - `top_int`: The top boundary coordinate of the cropped region. Type should be `INT`.
        - `left_int`: The left boundary coordinate of the cropped region. Type should be `INT`.
        - `right_int`: The right boundary coordinate of the cropped region. Type should be `INT`.
        - `bottom_int`: The bottom boundary coordinate of the cropped region. Type should be `INT`.
        - `width_int`: The width of the cropped region. Type should be `INT`.
        - `height_int`: The height of the cropped region. Type should be `INT`.