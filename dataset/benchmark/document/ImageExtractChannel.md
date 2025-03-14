- `ImageExtractChannel`: This node is designed to extract a specific channel (Red, Green, Blue, or Alpha) from a given set of images. It allows for the manipulation and analysis of individual color channels, which can be crucial for various image processing tasks, such as creating masks or isolating color components.
    - Inputs:
        - `images` (Required): The images from which a specific channel will be extracted. This input is crucial for determining the source images to be processed. Type should be `IMAGE`.
        - `channel` (Required): Specifies the color channel (Red, Green, Blue, or Alpha) to be extracted from the images. This choice directly affects the output by isolating the desired channel. Type should be `COMBO[STRING]`.
    - Outputs:
        - `channel_data`: The extracted channel data from the input images, provided as a mask. This output is useful for further image processing or analysis tasks. Type should be `MASK`.
