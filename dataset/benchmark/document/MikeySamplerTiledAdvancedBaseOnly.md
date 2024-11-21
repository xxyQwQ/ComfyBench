- `MikeySamplerTiledAdvancedBaseOnly`: This node is designed for advanced tiled sampling without the need for smooth steps or a refiner. It focuses on generating or processing images in a tiled manner, optimizing for scenarios where seamless integration of tiles is crucial without additional refinement steps.
    - Inputs:
        - `base_model` (Required): Specifies the base model used for sampling, setting the foundation for the generation process. Type should be `MODEL`.
        - `samples` (Required): The initial latent samples to be processed or refined through the sampling process. Type should be `LATENT`.
        - `vae` (Required): The variational autoencoder used alongside the base model to process or refine samples. Type should be `VAE`.
        - `positive_cond_base` (Required): Positive conditioning for the base model to guide the sampling towards desired attributes. Type should be `CONDITIONING`.
        - `negative_cond_base` (Required): Negative conditioning for the base model to steer the sampling away from undesired attributes. Type should be `CONDITIONING`.
        - `model_name` (Required): The name of the model, typically used to identify different models within a framework or library. Type should be `COMBO[STRING]`.
        - `seed` (Required): Seed for random number generation, ensuring reproducibility across sampling runs. Type should be `INT`.
        - `denoise_image` (Required): Specifies the degree of denoising applied to the image, affecting the clarity and quality of the output. Type should be `FLOAT`.
        - `steps` (Required): The number of steps to run in the sampling process, affecting the detail and quality of the generated image. Type should be `INT`.
        - `cfg` (Required): Controls the conditioning factor, influencing the generation's adherence to the given conditions. Type should be `FLOAT`.
        - `sampler_name` (Required): The name of the sampler used, affecting the diversity and quality of generated samples. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): The scheduler used to manage the sampling process, impacting the progression and variation of samples. Type should be `COMBO[STRING]`.
        - `upscale_by` (Required): The factor by which the image is upscaled, affecting the resolution and detail of the final output. Type should be `FLOAT`.
        - `tiler_denoise` (Required): Specifies the degree of denoising applied to each tile, affecting the consistency and quality of the tiled output. Type should be `FLOAT`.
        - `tile_size` (Required): Defines the size of each tile in the generated image, affecting the granularity of the output. Type should be `INT`.
        - `image_optional` (Optional): An optional image input that can be used to influence the sampling process. Type should be `IMAGE`.
    - Outputs:
        - `output_image`: The generated image after the tiled sampling process, reflecting the combined influence of all input parameters. Type should be `IMAGE`.