- `Join Image Lists`: The Join Image Lists node is designed to merge multiple lists of images into a single list, while also providing the sizes of the original lists. This functionality is essential for operations that require the consolidation of image data from various sources, ensuring seamless integration and manipulation of image collections.
    - Inputs:
        - `In1` (Required): Represents the first list of images to be joined. It is a required input that plays a crucial role in initiating the merging process. Type should be `IMAGE`.
        - `In2` (Required): Represents the second list of images to be joined. It is a required input that contributes to the merging process alongside the first list. Type should be `IMAGE`.
        - `In3` (Optional): An optional list of images that can be included in the merging process, providing flexibility in handling varying numbers of image lists. Type should be `IMAGE`.
        - `In4` (Optional): An optional list of images that can be included in the merging process, providing flexibility in handling varying numbers of image lists. Type should be `IMAGE`.
        - `In5` (Optional): An optional list of images that can be included in the merging process, providing flexibility in handling varying numbers of image lists. Type should be `IMAGE`.
    - Outputs:
        - `Joined`: The combined list of images from all input lists. Type should be `IMAGE`.
        - `Sizes`: A list of integers representing the sizes of the original input lists. Type should be `INT`.