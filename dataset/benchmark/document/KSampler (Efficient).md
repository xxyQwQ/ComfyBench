- `KSampler (Efficient)`: The KSampler (Efficient) node is designed to efficiently sample latent images using a variety of sampling strategies, including regular and advanced methods. It leverages the Comfy KSampler nodes to generate or refine latent images based on specified conditions, aiming to optimize the sampling process for speed and resource usage.
    - Inputs:
        - `model` (Required): Specifies the model to be used for sampling. It is crucial for determining the sampling behavior and the quality of the generated images. Type should be `MODEL`.
        - `seed` (Required): Determines the random seed for sampling, ensuring reproducibility of results. It affects the randomness of the generated latent images. Type should be `INT`.
        - `steps` (Required): Defines the number of steps to perform during sampling. It influences the detail and quality of the generated images. Type should be `INT`.
        - `cfg` (Required): Controls the conditioning-free guidance (CFG) scale, affecting the adherence to the specified conditions. Type should be `FLOAT`.
        - `sampler_name` (Required): Identifies the specific sampler algorithm to use, impacting the sampling method and results. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Specifies the scheduler for controlling the sampling process, affecting the progression of image generation. Type should be `COMBO[STRING]`.
        - `positive` (Required): The positive conditioning text to guide the image generation towards desired attributes. Type should be `CONDITIONING`.
        - `negative` (Required): The negative conditioning text to steer the image generation away from undesired attributes. Type should be `CONDITIONING`.
        - `latent_image` (Required): The initial latent image to refine or generate from, if provided. It influences the starting point of the sampling process. Type should be `LATENT`.
        - `denoise` (Required): Adjusts the level of denoising applied during sampling, affecting the clarity and quality of the generated images. Type should be `FLOAT`.
        - `preview_method` (Required): unknown Type should be `COMBO[STRING]`.
        - `vae_decode` (Required): unknown Type should be `COMBO[STRING]`.
        - `optional_vae` (Optional): unknown Type should be `VAE`.
        - `script` (Optional): unknown Type should be `SCRIPT`.
    - Outputs:
        - `MODEL`: unknown Type should be `MODEL`.
        - `CONDITIONING+`: Positive conditioning information that has been applied or generated during the sampling process. Type should be `CONDITIONING`.
        - `CONDITIONING-`: Negative conditioning information that has been applied or generated during the sampling process. Type should be `CONDITIONING`.
        - `LATENT`: The sampled or refined latent image, representing the result of the sampling process. Type should be `LATENT`.
        - `VAE`: VAE-related output, potentially including encoded or decoded information as part of the sampling process. Type should be `VAE`.
        - `IMAGE`: The final image output, representing the visual result of the sampling and conditioning process. Type should be `IMAGE`.
