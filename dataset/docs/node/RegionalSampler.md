- `RegionalSampler`: The RegionalSampler node is designed for advanced regional sampling within latent spaces, allowing for precise manipulation and enhancement of specific regions in generated images or data. This node facilitates targeted interventions in the sampling process, enabling users to apply different sampling strategies or adjustments to designated areas, thereby achieving more nuanced and customized generation outcomes.
    - Parameters:
        - `seed`: Specifies the initial seed for random number generation, influencing the sampling process's reproducibility and variability. Type should be `INT`.
        - `seed_ind`: Provides a secondary seed option to further customize the sampling process, offering additional control over the randomness and variation in the output. Type should be `INT`.
        - `seed_ind_mode`: Determines how the secondary seed is utilized in the sampling process, allowing for various modes of operation that can adjust the influence of the secondary seed on the overall sampling. Type should be `COMBO[STRING]`.
        - `steps`: Defines the number of steps to be taken in the sampling process, impacting the depth and detail of the sampling operation. Type should be `INT`.
        - `base_only_steps`: Specifies the number of initial steps that exclusively use the base sampler, setting the foundation for the subsequent regional sampling. Type should be `INT`.
        - `denoise`: Controls the level of denoising applied to the sampled output, affecting the clarity and quality of the generated results. Type should be `FLOAT`.
        - `overlap_factor`: Determines the degree of overlap between sampled regions, influencing the blending and transition between different sampling areas. Type should be `INT`.
        - `restore_latent`: A boolean flag indicating whether the original latent state should be restored after sampling, affecting the preservation of the initial input characteristics. Type should be `BOOLEAN`.
        - `additional_mode`: Specifies the mode of operation for additional sampling strategies, offering flexibility in how supplementary sampling is integrated. Type should be `COMBO[STRING]`.
        - `additional_sampler`: Selects an additional sampler to be used in conjunction with the base sampler, enabling a combination of sampling techniques for enhanced results. Type should be `COMBO[STRING]`.
        - `additional_sigma_ratio`: Sets the ratio of sigma values used in the additional sampler, affecting the intensity and impact of the supplementary sampling process. Type should be `FLOAT`.
    - Inputs:
        - `samples`: The input latent samples to be processed and manipulated through the regional sampling strategy. Type should be `LATENT`.
        - `base_sampler`: The base sampler used as the starting point for the regional sampling process, providing the initial sampling framework. Type should be `KSAMPLER_ADVANCED`.
        - `regional_prompts`: Defines specific prompts or conditions for the regional sampling, allowing for targeted adjustments and enhancements within designated areas. Type should be `REGIONAL_PROMPTS`.
    - Outputs:
        - `latent`: The modified latent samples after regional sampling, reflecting the targeted adjustments and enhancements applied to specific regions. Type should be `LATENT`.