- `ADE_NoiseLayerReplace`: This node is designed to replace specific noise layers within an animation or image generation process, allowing for the customization of noise patterns by overlaying new noise on top of existing noise based on a mask. It facilitates the creation of varied visual textures and effects by selectively modifying the noise characteristics.
    - Parameters:
        - `batch_offset`: Determines the offset for batch processing, affecting how noise patterns are applied across different items in a batch. Type should be `INT`.
        - `noise_type`: Specifies the type of noise to be used in the replacement process, influencing the visual texture and characteristics of the output. Type should be `COMBO[STRING]`.
        - `seed_gen_override`: Overrides the default seed generator, allowing for custom seed generation strategies that can affect the randomness and distribution of the noise. Type should be `COMBO[STRING]`.
        - `seed_offset`: Applies an additional offset to the seed value, further customizing the randomness of the noise generation. Type should be `INT`.
        - `seed_override`: Directly specifies a seed value, overriding any automatic seed generation for precise control over the noise pattern. Type should be `INT`.
    - Inputs:
        - `prev_noise_layers`: Specifies the previous configuration of noise layers to be modified, allowing for the integration of the new noise layer into the existing sequence. Type should be `NOISE_LAYERS`.
        - `mask_optional`: Defines a mask to control where the noise is replaced, enabling selective application of new noise patterns on specific areas. Type should be `MASK`.
    - Outputs:
        - `noise_layers`: Returns the modified noise layer configuration, incorporating the newly replaced noise layer for subsequent processing. Type should be `NOISE_LAYERS`.