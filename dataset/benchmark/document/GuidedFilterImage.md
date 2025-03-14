- `GuidedFilterImage`: The GuidedFilterImage node applies a guided filter to a batch of images using a guide image to enhance the output images' details while preserving edges. This process involves adjusting the images based on the spatial correlations indicated by the guide image, making it suitable for tasks like smoothing, detail enhancement, or noise reduction.
    - Inputs:
        - `images` (Required): The batch of images to be filtered. This input is crucial for determining the content that will undergo the guided filtering process. Type should be `IMAGE`.
        - `guide` (Required): The guide image used to direct the filtering process. It plays a key role in determining how the filtering is applied across different regions of the input images. Type should be `IMAGE`.
        - `size` (Required): Specifies the size of the kernel used for the guided filter. A larger size can lead to more pronounced smoothing and edge preservation. Type should be `INT`.
        - `sigma` (Required): Controls the degree of smoothing in the guided filtering process. Higher values result in stronger smoothing effects. Type should be `FLOAT`.
    - Outputs:
        - `image`: The filtered images, which have undergone guided filtering to enhance details while preserving edges. Type should be `IMAGE`.
