- `Image Crop Location`: This node is designed to crop a given image based on specified coordinates and dimensions, adjusting the crop area to fit within the original image's boundaries. It ensures the cropped area does not exceed the image's dimensions by dynamically adjusting the specified coordinates and size.
    - Inputs:
        - `image` (Required): The input image to be cropped. This parameter is crucial as it defines the base image from which a specific region will be extracted based on the provided coordinates. Type should be `IMAGE`.
        - `top` (Required): The top coordinate for the cropping area, influencing the vertical starting point of the crop within the original image. Type should be `INT`.
        - `left` (Required): The left coordinate for the cropping area, influencing the horizontal starting point of the crop within the original image. Type should be `INT`.
        - `right` (Required): The right coordinate for the cropping area, determining the horizontal end point of the crop within the original image. Type should be `INT`.
        - `bottom` (Required): The bottom coordinate for the cropping area, determining the vertical end point of the crop within the original image. Type should be `INT`.
    - Outputs:
        - `image`: The cropped image. Type should be `IMAGE`.
        - `crop_data`: Crop data including the final crop dimensions and coordinates. Type should be `CROP_DATA`.
