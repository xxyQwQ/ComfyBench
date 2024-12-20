- `Latent Upscale by Factor (WAS)`: This node specializes in upscaling latent images by a specified factor, utilizing various interpolation methods to enhance the resolution while maintaining the integrity of the original image. It provides flexibility in scaling and alignment options to cater to different upscaling needs.
    - Inputs:
        - `samples` (Required): The latent images to be upscaled. This input is crucial as it determines the base content that will undergo the upscaling process. Type should be `LATENT`.
        - `mode` (Required): Specifies the interpolation method to be used for upscaling, offering options like 'area', 'bicubic', 'bilinear', and 'nearest'. This choice affects the quality and characteristics of the upscaled image. Type should be `COMBO[STRING]`.
        - `factor` (Required): The scaling factor by which the latent images will be upscaled. A positive float that directly influences the final size of the upscaled images. Type should be `FLOAT`.
        - `align` (Required): A boolean flag indicating whether to align corners in certain interpolation modes, affecting the upscaling precision and outcome. Type should be `COMBO[STRING]`.
    - Outputs:
        - `latent`: The upscaled latent images, enhanced in resolution according to the specified factor and interpolation method. Type should be `LATENT`.
