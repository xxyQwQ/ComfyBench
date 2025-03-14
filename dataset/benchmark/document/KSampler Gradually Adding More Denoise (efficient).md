- `KSampler Gradually Adding More Denoise (efficient)`: This node specializes in generating a sequence of samples by gradually increasing the denoising strength applied to a latent image. It leverages a common sampling function to produce a series of progressively denoised images, allowing for a controlled and incremental refinement of the generated content.
    - Inputs:
        - `model` (Required): The model parameter represents the neural network model used for the sampling process. It is crucial for determining the behavior and quality of the generated samples. Type should be `MODEL`.
        - `positive` (Required): Positive conditioning information that guides the generation towards desired attributes or features in the output. Type should be `CONDITIONING`.
        - `negative` (Required): Negative conditioning information used to steer the generation away from certain attributes or features, complementing the positive conditioning. Type should be `CONDITIONING`.
        - `latent_image` (Required): The initial latent representation of the image to be denoised. This serves as the starting point for the gradual denoising process. Type should be `LATENT`.
        - `seed` (Required): A seed value for random number generation, ensuring reproducibility of the sampling process. Type should be `INT`.
        - `steps` (Required): The number of steps to take in the sampling process, affecting the granularity of the denoising progression. Type should be `INT`.
        - `cfg` (Required): The CFG (Classifier-Free Guidance) scale, which adjusts the influence of conditioning information on the generation process. Type should be `FLOAT`.
        - `sampler_name` (Required): The name of the sampler algorithm to use, which determines the specific approach to sampling and denoising. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): The scheduler determines the sequence of noise levels used throughout the sampling process, influencing the progression of denoising. Type should be `COMBO[STRING]`.
        - `start_denoise` (Required): The initial denoising strength, setting the starting point for the gradual increase in denoising. Type should be `FLOAT`.
        - `denoise_increment` (Required): The amount by which the denoising strength is increased at each step, controlling the pace of the denoising progression. Type should be `FLOAT`.
        - `denoise_increment_steps` (Required): The total number of steps over which the denoising strength is increased, defining the length of the gradual denoising process. Type should be `INT`.
        - `optional_vae` (Optional): An optional VAE (Variational Autoencoder) model that can be used in conjunction with the primary model for additional processing or refinement of the generated samples. Type should be `VAE`.
    - Outputs:
        - `MODEL`: The neural network model used for the sampling process, returned unchanged. Type should be `MODEL`.
        - `CONDITIONING+`: The positive conditioning information, returned unchanged. Type should be `CONDITIONING`.
        - `CONDITIONING-`: The negative conditioning information, returned unchanged. Type should be `CONDITIONING`.
        - `LATENT`: The final latent representation of the image after the gradual denoising process. Type should be `LATENT`.
        - `VAE`: The optional VAE model, if used, returned unchanged. Type should be `VAE`.
