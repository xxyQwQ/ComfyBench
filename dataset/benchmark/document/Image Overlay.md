- `Image Overlay`: The ImageOverlay node is designed to overlay one image onto another with various customization options such as resizing, rotation, opacity adjustment, and optional masking. This node enables the creation of composite images by applying transformations and blending techniques to the overlay image before merging it with the base image.
    - Inputs:
        - `base_image` (Required): The base image onto which the overlay image will be applied. It serves as the background for the composite image. Type should be `IMAGE`.
        - `overlay_image` (Required): The image to be overlaid onto the base image. This image can be resized, rotated, and its opacity adjusted before overlaying. Type should be `IMAGE`.
        - `overlay_resize` (Required): Specifies how the overlay image should be resized to fit the base image, with options including fitting to dimensions, rescaling by a factor, or resizing to specific width and height. Type should be `COMBO[STRING]`.
        - `resize_method` (Required): The method used for resizing the overlay image, affecting the quality and appearance of the resized image. Type should be `COMBO[STRING]`.
        - `rescale_factor` (Required): The factor by which the overlay image is rescaled when the 'Resize by rescale_factor' option is selected. Type should be `FLOAT`.
        - `width` (Required): The width to which the overlay image is resized when the 'Resize to width & height' option is selected. Type should be `INT`.
        - `height` (Required): The height to which the overlay image is resized when the 'Resize to width & height' option is selected. Type should be `INT`.
        - `x_offset` (Required): The horizontal offset at which the overlay image is placed over the base image. Type should be `INT`.
        - `y_offset` (Required): The vertical offset at which the overlay image is placed over the base image. Type should be `INT`.
        - `rotation` (Required): The angle in degrees to rotate the overlay image before applying it to the base image. Type should be `INT`.
        - `opacity` (Required): The opacity level of the overlay image, allowing for transparent overlays. Type should be `FLOAT`.
        - `optional_mask` (Optional): An optional mask that can be applied to the overlay image, controlling the transparency of different parts of the overlay. Type should be `MASK`.
    - Outputs:
        - `image`: The base image after the overlay has been applied, reflecting all specified transformations and adjustments. Type should be `IMAGE`.