- `ImageMuxer`: The ImageMuxer node is designed to select and output one image from a set of up to four input images based on a given selector index. This functionality is essential for scenarios where dynamic image selection is required, such as in image processing pipelines or conditional image rendering tasks.
    - Inputs:
        - `image_i` (Required): Represents one of the up to four image inputs for selection, acting as potential outputs based on the selector index. This generalization covers all image inputs, allowing for dynamic selection from multiple sources. Type should be `IMAGE`.
        - `input_selector` (Required): An integer index that determines which of the input images is selected for output. This selector drives the dynamic selection process. Type should be `INT`.
    - Outputs:
        - `image`: The selected image based on the input selector index. This output facilitates dynamic image selection in various applications. Type should be `IMAGE`.
