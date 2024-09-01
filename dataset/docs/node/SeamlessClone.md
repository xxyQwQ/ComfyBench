- `SeamlessClone`: The SeamlessClone node is designed for blending two images together in a seamless manner, using a mask to define the blending region and various cloning modes to control the blending behavior. It allows for sophisticated image compositing techniques, such as normal cloning, mixed cloning, and monochrome transfer.
    - Parameters:
        - `flag`: Specifies the cloning mode to be used for the operation, such as normal, mixed, or monochrome transfer. This affects how the source image is blended into the destination. Type should be `COMBO[STRING]`.
        - `cx`: The x-coordinate of the center of the destination point for cloning. It influences the positioning of the cloned region. Type should be `INT`.
        - `cy`: The y-coordinate of the center of the destination point for cloning. It influences the positioning of the cloned region. Type should be `INT`.
    - Inputs:
        - `dst`: The destination image onto which the source image will be cloned. It serves as the backdrop for the operation. Type should be `IMAGE`.
        - `src`: The source image to be cloned onto the destination image. This image will be blended into the destination based on the mask and cloning mode. Type should be `IMAGE`.
        - `src_mask`: A mask image defining the region of the source image to be cloned. Only the parts of the source image covered by the mask will be cloned. Type should be `IMAGE`.
    - Outputs:
        - `image`: The result of the seamless cloning operation, which is an image with the source region blended into the destination based on the specified mask and mode. Type should be `IMAGE`.