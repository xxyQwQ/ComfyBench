- `RegionalSeedExplorerMask __Inspire`: The RegionalSeedExplorerMask node is designed to apply variations to noise patterns based on seed prompts and masks, enabling the exploration of diverse visual outcomes within specified regions. It leverages masks to focus changes, allowing for targeted modifications and enhancements in image generation processes.
    - Parameters:
        - `seed_prompt`: The seed_prompt parameter contains the seed prompts that define the variations to be applied to the noise pattern, guiding the generation of specific visual outcomes. Type should be `STRING`.
        - `enable_additional`: This parameter controls whether additional seed prompts and their corresponding strengths are included in the variation process, allowing for more complex modifications. Type should be `BOOLEAN`.
        - `additional_seed`: When additional modifications are enabled, this parameter specifies the additional seed prompt to be applied. Type should be `INT`.
        - `additional_strength`: This parameter determines the strength of the additional seed prompt's effect on the noise pattern, allowing for fine-tuned adjustments. Type should be `FLOAT`.
        - `noise_mode`: The noise_mode parameter specifies whether the processing should be performed on the CPU or GPU, affecting performance and resource utilization. Type should be `COMBO[STRING]`.
    - Inputs:
        - `mask`: The mask parameter specifies the region within the noise pattern where the seed prompt variations will be applied, enabling targeted modifications. Type should be `MASK`.
        - `noise`: The noise parameter represents the initial noise pattern to which the seed prompt variations will be applied, serving as the base for generating diverse visual outcomes. Type should be `NOISE`.
    - Outputs:
        - `noise`: The modified noise pattern, reflecting the applied seed prompt variations within the specified mask region. Type should be `NOISE`.