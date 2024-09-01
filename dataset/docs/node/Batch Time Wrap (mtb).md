- `Batch Time Wrap (mtb)`: The Batch Time Wrap (mtb) node is designed to remap a batch of images according to a specified time curve, effectively allowing for the dynamic adjustment of image sequences based on temporal data.
    - Parameters:
        - `target_count`: Specifies the desired number of images in the output batch, allowing for dynamic adjustment of the batch size. Type should be `INT`.
    - Inputs:
        - `frames`: The input batch of images to be remapped according to the time curve. Type should be `IMAGE`.
        - `curve`: A sequence of floating-point values defining the time curve along which the input images are remapped. Type should be `FLOATS`.
    - Outputs:
        - `image`: The output batch of images that have been remapped according to the specified time curve. Type should be `IMAGE`.
        - `interpolated_floats`: A sequence of floating-point values representing the interpolated positions of the input images along the time curve. Type should be `FLOATS`.