- `Calculate Upscale`: The Calculate Upscale node is designed to compute the necessary upscale factor and tile size for an image given a target height and the number of tiles in the horizontal direction. It aims to facilitate image processing tasks by determining how much an image needs to be upscaled to meet specific dimensions, thereby assisting in optimizing image manipulation and rendering processes.
    - Inputs:
        - `image` (Required): The image parameter represents the input image to be processed. It is crucial for determining the current dimensions of the image, which are necessary to calculate the upscale factor and tile size. Type should be `IMAGE`.
        - `target_height` (Required): The target_height parameter specifies the desired height of the image after upscaling. It plays a key role in calculating the upscale factor by comparing it with the current height of the image. Type should be `INT`.
        - `tiles_in_x` (Required): The tiles_in_x parameter indicates the number of horizontal tiles the upscaled image should be divided into. This is essential for calculating the width of each tile after the image has been upscaled. Type should be `INT`.
    - Outputs:
        - `tile_size`: The tile_size output represents the width of each tile in the upscaled image. It is calculated based on the upscaled width of the image and the number of tiles in the horizontal direction. Type should be `INT`.
        - `upscale`: The upscale output indicates the factor by which the image needs to be upscaled to achieve the target height. It is a crucial metric for resizing images in image processing tasks. Type should be `FLOAT`.