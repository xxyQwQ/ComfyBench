- `Image to Latent Mask`: This node is designed to convert an image into a latent mask representation, facilitating the manipulation and analysis of images in a latent space. It abstracts the complexities of image processing and latent space conversion, providing a streamlined approach to working with image data in advanced computational contexts.
    - Inputs:
        - `images` (Required): The 'images' parameter represents the input images to be converted into a latent mask. This conversion is crucial for enabling further operations in the latent space, such as blending or compositing with other latent representations. Type should be `IMAGE`.
        - `channel` (Required): The 'channel' parameter specifies the color channel ('red', 'green', 'blue', 'alpha') of the input image to be used in the mask creation process. This selection can significantly influence the resulting latent mask by highlighting specific features or areas of interest. Type should be `COMBO[STRING]`.
    - Outputs:
        - `MASKS`: The output is a collection of masks derived from the specified image channels. These masks can be utilized in various image manipulation and analysis tasks within a latent space framework. Type should be `MASK`.