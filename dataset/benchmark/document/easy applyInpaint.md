- `easy applyInpaint`: This node facilitates the application of various inpainting techniques to images, offering a flexible approach to restoring or modifying specific areas of an image based on the inpainting mode selected. It integrates seamlessly with different inpainting strategies, such as 'fooocus_inpaint', 'brushnet_random', and others, to accommodate diverse inpainting needs.
    - Inputs:
        - `pipe` (Required): Represents the pipeline configuration, serving as the foundational structure for the inpainting process. Type should be `PIPE_LINE`.
        - `image` (Required): The image to be inpainted, acting as the primary input for the inpainting operation. Type should be `IMAGE`.
        - `mask` (Required): A mask that specifies the areas of the image to be inpainted, guiding the inpainting algorithm on where to apply its effects. Type should be `MASK`.
        - `inpaint_mode` (Required): Determines the inpainting strategy to be used, offering options like 'normal', 'fooocus_inpaint', 'brushnet_random', and others to suit various inpainting scenarios. Type should be `['normal', 'fooocus_inpaint', 'brushnet_random', 'brushnet_segmentation', 'powerpaint']`.
        - `encode` (Required): Specifies the encoding method to prepare the image for inpainting, with options including 'none', 'vae_encode_inpaint', and others, affecting the preprocessing step. Type should be `['none', 'vae_encode_inpaint', 'inpaint_model_conditioning', 'different_diffusion']`.
        - `grow_mask_by` (Required): Defines how much to expand the mask beyond its original boundaries, which can influence the inpainting results by altering the area considered for inpainting. Type should be `INT`.
        - `dtype` (Required): Specifies the data type for the computation, affecting performance and precision. Options include 'float16', 'bfloat16', 'float32', 'float64'. Type should be `COMBO[STRING]`.
        - `fitting` (Required): Adjusts the fitting level of the inpainting to the surrounding image area, influencing how seamlessly the inpainted area blends with the rest of the image. Type should be `FLOAT`.
        - `function` (Required): Selects the specific inpainting function to use, such as 'text guided', 'shape guided', or 'object removal', tailoring the inpainting process to the content of the image. Type should be `COMBO[STRING]`.
        - `scale` (Required): Controls the scale of the inpainting effect, which can affect the intensity and visibility of the inpainting within the specified area. Type should be `FLOAT`.
        - `start_at` (Required): Determines the starting point of the inpainting process within the image, allowing for precise control over the area of effect. Type should be `INT`.
        - `end_at` (Required): Specifies the ending point of the inpainting process, defining the scope of the area to be inpainted. Type should be `INT`.
    - Outputs:
        - `pipe`: The modified pipeline configuration after applying the inpainting process, reflecting any changes made to the image. Type should be `PIPE_LINE`.
