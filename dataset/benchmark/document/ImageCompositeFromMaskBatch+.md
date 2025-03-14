- `ImageCompositeFromMaskBatch+`: This node is designed for image manipulation tasks, specifically for creating composite images from two source images based on a mask. It blends parts of the 'image_from' and 'image_to' images according to the mask, allowing for sophisticated image editing and composition techniques.
    - Inputs:
        - `image_from` (Required): The source image from which pixels are taken when the mask is not applied. It plays a crucial role in determining the final composite image's appearance. Type should be `IMAGE`.
        - `image_to` (Required): The target image to which pixels are added based on the mask. It significantly influences the outcome of the composite image. Type should be `IMAGE`.
        - `mask` (Required): A binary or grayscale mask that determines how pixels from 'image_from' and 'image_to' are blended together. The mask's values dictate the blending process, affecting the composite image's visual result. Type should be `MASK`.
    - Outputs:
        - `image`: The resulting composite image, created by blending 'image_from' and 'image_to' according to the mask. It showcases the combined visual elements of both source images. Type should be `IMAGE`.
