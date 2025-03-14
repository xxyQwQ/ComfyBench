- `LayerUtility_ LaMa`: The LaMa node utilizes the LaMa (Large Mask inpainting) model for sophisticated image inpainting tasks. It excels in removing unwanted elements from images or filling in missing areas, offering high-quality restoration and manipulation capabilities.
    - Inputs:
        - `image` (Required): The input image to be processed for inpainting, serving as the primary subject for object removal or area filling. Type should be `IMAGE`.
        - `mask` (Required): A binary mask indicating the areas to inpaint within the image, guiding the model on where to apply its restoration efforts. Type should be `MASK`.
        - `lama_model` (Required): Specifies the LaMa model to be used for the inpainting task, allowing for customization of the inpainting process. Type should be `COMBO[STRING]`.
        - `device` (Required): The computing device (CPU/GPU) on which the inpainting operation is performed, affecting performance and efficiency. Type should be `COMBO[STRING]`.
        - `invert_mask` (Required): A boolean indicating whether to invert the inpainting mask, altering the areas targeted for inpainting. Type should be `BOOLEAN`.
        - `mask_grow` (Required): An integer specifying how much to expand the inpainting mask, enabling broader area coverage. Type should be `INT`.
        - `mask_blur` (Required): An integer defining the level of blur applied to the mask edges, smoothing the transition between inpainted and original areas. Type should be `INT`.
    - Outputs:
        - `image`: The output image after inpainting, showcasing the areas where unwanted elements have been removed or missing parts filled in. Type should be `IMAGE`.
