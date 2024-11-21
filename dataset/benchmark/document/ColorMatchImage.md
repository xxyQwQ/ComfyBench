- `ColorMatchImage`: The ColorMatchImage node is designed for advanced image processing, specifically focusing on matching the color distribution of one image to another. This process is essential for tasks that require the integration of images from different sources to ensure visual consistency across them. The node offers options for selecting the type and intensity of the blur applied during the color matching process, allowing for flexible adaptation to various image characteristics.
    - Inputs:
        - `images` (Required): The input images whose color distribution needs to be matched to the reference image. This parameter is crucial for defining the source images on which color matching will be performed. Type should be `IMAGE`.
        - `reference` (Required): The reference image to which the color distribution of the input images will be matched. It serves as the target for color matching, guiding the adjustment process. Type should be `IMAGE`.
        - `blur_type` (Required): Specifies the type of blur to be applied during the color matching process, such as Gaussian blur or guided filter, affecting the smoothness and detail preservation. Type should be `COMBO[STRING]`.
        - `blur_size` (Required): The size of the blur filter to be applied, determining the extent of blurring and its impact on the color matching process. Type should be `INT`.
        - `factor` (Required): A factor controlling the intensity of the color matching effect, allowing for fine-tuning the balance between the original and matched colors. Type should be `FLOAT`.
    - Outputs:
        - `image`: The output image with its color distribution matched to the reference image, integrating seamlessly with the target visual context. Type should be `IMAGE`.