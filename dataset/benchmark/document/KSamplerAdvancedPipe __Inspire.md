- `KSamplerAdvancedPipe __Inspire`: This node is designed for advanced sampling in generative models, focusing on creating or modifying latent images through a comprehensive set of parameters. It integrates various components such as noise addition, seed manipulation, and conditioning adjustments to refine the generation process, aiming to produce high-quality, customizable results.
    - Inputs:
        - `basic_pipe` (Required): A tuple containing the model, clip, VAE, and positive and negative conditioning components, serving as the foundational setup for the sampling process. Type should be `BASIC_PIPE`.
        - `add_noise` (Required): A boolean flag indicating whether noise should be added to the generation process, affecting the diversity and quality of the output. Type should be `BOOLEAN`.
        - `noise_seed` (Required): An integer seed specifically for noise generation, contributing to the variability and uniqueness of the generated images. Type should be `INT`.
        - `steps` (Required): The number of steps to perform in the sampling process, affecting the detail and quality of the generated images. Type should be `INT`.
        - `cfg` (Required): A configuration parameter that influences the sampling behavior, potentially affecting the style and characteristics of the output. Type should be `FLOAT`.
        - `sampler_name` (Required): The name of the sampler to use, determining the specific sampling algorithm applied. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Specifies the scheduler for controlling the sampling process, affecting the progression of image generation. Type should be `COMBO[STRING]`.
        - `latent_image` (Required): The initial latent image to be modified or enhanced through the sampling process. Type should be `LATENT`.
        - `start_at_step` (Required): The step at which to start the sampling process, allowing for control over the generation's initiation point. Type should be `INT`.
        - `end_at_step` (Required): The final step of the sampling process, defining the endpoint of image generation. Type should be `INT`.
        - `noise_mode` (Required): Determines the mode of noise application, influencing the texture and quality of the output. Type should be `COMBO[STRING]`.
        - `return_with_leftover_noise` (Required): A flag indicating whether to return the image with any residual noise, affecting the final image's appearance. Type should be `BOOLEAN`.
        - `batch_seed_mode` (Required): Specifies the mode for seed generation in batch processing, impacting the diversity of generated images. Type should be `COMBO[STRING]`.
        - `variation_seed` (Required): An optional seed for introducing variations, allowing for controlled randomness in the output. Type should be `INT`.
        - `variation_strength` (Required): Determines the strength of the applied variations, affecting the degree of change from the original image. Type should be `FLOAT`.
        - `noise_opt` (Optional): Optional noise parameters for further customization of the noise application process. Type should be `NOISE`.
        - `scheduler_func_opt` (Optional): An optional scheduler function for advanced control over the sampling schedule. Type should be `SCHEDULER_FUNC`.
    - Outputs:
        - `latent`: Represents the generated or modified latent image, showcasing the effectiveness of the sampling process. Type should be `LATENT`.
        - `vae`: The VAE component used in the sampling process, potentially modified or utilized as part of the generation. Type should be `VAE`.