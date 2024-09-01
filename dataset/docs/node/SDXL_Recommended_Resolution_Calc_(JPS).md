- `SDXL Recommended Resolution Calc (JPS)`: This node calculates the recommended resolution for SDXL (Stable Diffusion XL) based on a target width and height, aiming to find the closest matching aspect ratio from a predefined set. It considers horizontal, vertical, and square aspect ratios to determine the most suitable resolution for generating images with the desired dimensions.
    - Parameters:
        - `target_width`: Specifies the desired width of the image. It plays a crucial role in determining the closest matching aspect ratio for the recommended resolution. Type should be `INT`.
        - `target_height`: Specifies the desired height of the image. It is used alongside the target width to calculate the closest matching aspect ratio for the recommended resolution. Type should be `INT`.
    - Inputs:
    - Outputs:
        - `SDXL_width`: The recommended width for the SDXL image, based on the closest matching aspect ratio. Type should be `INT`.
        - `SDXL_height`: The recommended height for the SDXL image, based on the closest matching aspect ratio. Type should be `INT`.