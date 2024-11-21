- `BatchCLIPSeg`: This node is designed for batch processing of images using the CLIPSeg model for image segmentation. It leverages the CLIPSegForImageSegmentation and CLIPSegProcessor from the transformers library to perform semantic segmentation on a collection of images, adapting the model and processor to the specific hardware configuration (CPU or GPU) and data type for efficient execution.
    - Inputs:
        - `images` (Required): The input images to be processed for segmentation. This parameter is crucial for the node's operation as it directly affects the segmentation results. Type should be `IMAGE`.
        - `text` (Required): The text prompt used to guide the segmentation process, influencing the areas of the image that will be segmented. Type should be `STRING`.
        - `threshold` (Required): A threshold value for determining the segmentation cut-off, affecting the sensitivity of the segmentation process. Type should be `FLOAT`.
        - `binary_mask` (Required): A boolean flag indicating whether the output mask should be binary, affecting the format of the segmentation result. Type should be `BOOLEAN`.
        - `combine_mask` (Required): A boolean flag indicating whether to combine the output masks for batch processing, affecting the structure of the output. Type should be `BOOLEAN`.
        - `use_cuda` (Required): A flag indicating whether to use CUDA for processing. This affects the execution speed and efficiency, especially for large batches of images. Type should be `BOOLEAN`.
        - `blur_sigma` (Optional): The sigma value for Gaussian blur to apply to the output mask, affecting the smoothness of the mask edges. Type should be `FLOAT`.
        - `opt_model` (Optional): An optional pre-loaded model and processor to use for segmentation, allowing for flexibility in model choice and potential reuse of resources. Type should be `CLIPSEGMODEL`.
        - `prev_mask` (Optional): An optional previous mask to be combined with the current segmentation, allowing for iterative segmentation refinement. Type should be `MASK`.
        - `image_bg_level` (Optional): The background level for the image, affecting the contrast and visibility of the segmentation. Type should be `FLOAT`.
        - `invert` (Optional): A boolean flag indicating whether to invert the output mask, affecting the segmentation's focus area. Type should be `BOOLEAN`.
    - Outputs:
        - `Mask`: The output mask generated from the segmentation process, providing a binary or probabilistic map of the segmented areas. Type should be `MASK`.
        - `Image`: The original image overlaid with the segmentation mask, visually representing the segmentation results. Type should be `IMAGE`.