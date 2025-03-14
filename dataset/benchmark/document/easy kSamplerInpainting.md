- `easy kSamplerInpainting`: This node specializes in generating images by inpainting missing parts or modifying existing images. It leverages a variety of inpainting models and techniques to fill in gaps or replace sections of an image based on the surrounding context, allowing for creative and customized image manipulation.
    - Inputs:
        - `pipe` (Required): Represents the pipeline configuration for the inpainting process, including the model and parameters to be used. Type should be `PIPE_LINE`.
        - `grow_mask_by` (Required): Specifies how much to expand the inpainting mask by, in pixels, to ensure smoother transitions and better inpainting results. Type should be `INT`.
        - `image_output` (Required): Determines how the output image is handled, offering options for hiding, previewing, saving, or sending the image. Type should be `COMBO[STRING]`.
        - `link_id` (Required): An identifier for linking this operation with other processes or outputs, facilitating complex workflows. Type should be `INT`.
        - `save_prefix` (Required): A prefix for naming saved images, allowing for organized output management. Type should be `STRING`.
        - `additional` (Required): Selects the specific inpainting model or technique to be used, providing flexibility in approach and customization of the inpainting effect. Type should be `COMBO[STRING]`.
        - `model` (Optional): The inpainting model used for generating the output, integral to the inpainting process. Type should be `MODEL`.
        - `mask` (Optional): The mask that defines the area to be inpainted, crucial for guiding the inpainting process. Type should be `MASK`.
    - Outputs:
        - `pipe`: The updated pipeline configuration after the inpainting process, including any modifications to the model or parameters. Type should be `PIPE_LINE`.
        - `image`: The resulting image after inpainting, showcasing the modifications or filled gaps. Type should be `IMAGE`.
        - `vae`: The variational autoencoder used in the inpainting process, if applicable, indicating its role in generating or refining the output. Type should be `VAE`.
