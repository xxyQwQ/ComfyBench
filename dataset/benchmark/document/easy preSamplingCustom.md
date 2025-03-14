- `easy preSamplingCustom`: This node is designed for custom pre-sampling settings in image generation pipelines, allowing users to fine-tune the sampling process by specifying various parameters such as guidance, configuration scale, and noise levels. It provides a flexible interface for adjusting the pre-sampling behavior to achieve desired image qualities or effects.
    - Inputs:
        - `pipe` (Required): Represents the pipeline configuration, including model, positive and negative prompts, and other settings, serving as the foundation for the pre-sampling process. Type should be `PIPE_LINE`.
        - `guider` (Required): Specifies the guidance mode to be used during sampling, offering options like CFG, DualCFG, and others to influence the direction of the generated content. Type should be `COMBO[STRING]`.
        - `cfg` (Required): Defines the configuration scale, a floating-point value that adjusts the influence of the conditioning on the generated image, allowing for finer control over the sampling process. Type should be `FLOAT`.
        - `cfg_negative` (Required): Sets the negative configuration scale, a floating-point value that inversely influences the conditioning, providing a means to suppress undesired aspects in the generated content. Type should be `FLOAT`.
        - `sampler_name` (Required): Identifies the specific sampler to be used in the pre-sampling process, affecting the method of image generation. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Determines the scheduling algorithm for the sampling process, influencing the progression of steps in image generation. Type should be `COMBO[STRING]`.
        - `coeff` (Required): A parameter influencing the balance between different aspects of the sampling process, such as detail and coherence. Type should be `FLOAT`.
        - `steps` (Required): Specifies the number of steps to be taken in the sampling process, directly impacting the detail and quality of the generated image. Type should be `INT`.
        - `sigma_max` (Required): The maximum value of sigma for noise application, affecting the initial stages of the sampling process. Type should be `FLOAT`.
        - `sigma_min` (Required): The minimum value of sigma for noise application, affecting the final stages of the sampling process. Type should be `FLOAT`.
        - `rho` (Required): A parameter related to the noise schedule, influencing the distribution and impact of noise throughout the sampling process. Type should be `FLOAT`.
        - `beta_d` (Required): Determines the rate of diffusion in the sampling process, impacting the smoothness and blending of generated features. Type should be `FLOAT`.
        - `beta_min` (Required): The minimum value of beta for diffusion, affecting the detail and sharpness of the generated image. Type should be `FLOAT`.
        - `eps_s` (Required): A parameter controlling the epsilon scaling in the sampling process, influencing the variance of generated features. Type should be `FLOAT`.
        - `flip_sigmas` (Required): A boolean indicating whether to invert the sigma values, affecting the order and impact of noise application. Type should be `BOOLEAN`.
        - `denoise` (Required): Specifies the level of denoising to be applied, affecting the clarity and noise level of the generated image. Type should be `FLOAT`.
        - `add_noise` (Required): A boolean indicating whether additional noise should be added during the sampling process, affecting the texture and detail of the generated image. Type should be `COMBO[STRING]`.
        - `seed` (Required): Sets the seed for random number generation, ensuring reproducibility of the sampling process. Type should be `INT`.
        - `image_to_latent` (Optional): An optional parameter allowing for the conversion of an image to a latent representation, influencing the starting point of the sampling process. Type should be `IMAGE`.
        - `latent` (Optional): Specifies a latent image to be used as the starting point for the sampling process, affecting the initial state and direction of generation. Type should be `LATENT`.
        - `optional_sampler` (Optional): Allows for the selection of an alternative sampler, providing flexibility in the choice of sampling methods. Type should be `SAMPLER`.
        - `optional_sigmas` (Optional): Enables the specification of custom sigma values, offering control over the noise schedule and its impact on the sampling process. Type should be `SIGMAS`.
    - Outputs:
        - `pipe`: Outputs the modified pipeline configuration, reflecting the adjustments made during the pre-sampling process. Type should be `PIPE_LINE`.
