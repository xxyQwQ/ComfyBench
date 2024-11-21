- `ImpactKSamplerBasicPipe`: The ImpactKSamplerBasicPipe node is designed for advanced sampling techniques within the ComfyUI Impact Pack. It leverages a basic sampling pipeline to generate or modify latent images based on specified parameters, such as noise addition and classifier guidance, to achieve desired visual effects or enhancements.
    - Inputs:
        - `basic_pipe` (Required): Represents the core components required for sampling, including models and configurations necessary for the operation. Type should be `BASIC_PIPE`.
        - `seed` (Required): Used to initialize the random number generator for reproducible noise generation during sampling. Type should be `INT`.
        - `steps` (Required): Defines the total number of steps to perform during the sampling process. Type should be `INT`.
        - `cfg` (Required): Specifies the classifier free guidance value, influencing the direction and strength of the generated modifications. Type should be `FLOAT`.
        - `sampler_name` (Required): Identifies the specific sampler to use, allowing for different sampling strategies. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Determines the noise schedule to be applied throughout the sampling steps. Type should be `COMBO[STRING]`.
        - `latent_image` (Required): The input latent image to be processed or modified through sampling. Type should be `LATENT`.
        - `denoise` (Required): Controls the amount of noise reduction, affecting the final output's clarity and detail. Type should be `FLOAT`.
        - `scheduler_func_opt` (Optional): Optional parameter for providing a custom noise schedule function, offering more control over the sampling process. Type should be `SCHEDULER_FUNC`.
    - Outputs:
        - `basic_pipe`: Passes through the input basic_pipe, including all its components, unchanged. Type should be `BASIC_PIPE`.
        - `latent`: The resulting latent image after applying the sampling process, reflecting the specified modifications. Type should be `LATENT`.
        - `vae`: The VAE component from the input basic_pipe, included in the output for potential further processing. Type should be `VAE`.