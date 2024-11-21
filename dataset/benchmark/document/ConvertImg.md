- `ConvertImg`: The ConvertImg node is designed for explicit image format conversion within a custom node environment, facilitating direct transformations between different image color spaces without resorting to workarounds.
    - Inputs:
        - `image` (Required): The 'image' parameter represents the input image to be converted. Its role is crucial as it serves as the source image for the conversion process. Type should be `IMAGE`.
        - `to` (Required): The 'to' parameter specifies the target color space format for the conversion, influencing the output image's color representation. Type should be `COMBO[STRING]`.
    - Outputs:
        - `image`: The output is an image that has been converted to the specified color space format, reflecting the changes in color representation as per the 'to' parameter. Type should be `IMAGE`.