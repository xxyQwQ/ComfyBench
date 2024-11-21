- `NNLatentUpscale`: The NNLatentUpscale node specializes in enhancing the resolution of SDXL latent representations using a neural network approach. It dynamically adjusts its internal model based on the specified version and upscale factor, ensuring optimized upscaling of latent samples to achieve higher fidelity outputs.
    - Inputs:
        - `latent` (Required): The latent representation to be upscaled. It is crucial for defining the starting point of the upscaling process. Type should be `LATENT`.
        - `version` (Required): Specifies the version of the model to use for upscaling, allowing for flexibility and optimization based on the latent's characteristics. Type should be `COMBO[STRING]`.
        - `upscale` (Required): The factor by which to upscale the latent representation, directly influencing the output's resolution and detail. Type should be `FLOAT`.
    - Outputs:
        - `latent`: The upscaled latent representation, enhanced in resolution and detail. Type should be `LATENT`.