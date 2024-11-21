- `RegionalSeedExplorerColorMask __Inspire`: This node specializes in exploring seed variations within specific regions of an image, guided by a color mask. It applies modifications to the noise pattern based on seed prompts and additional parameters, allowing for targeted adjustments and enhancements in image generation processes.
    - Inputs:
        - `color_mask` (Required): The color mask used to define regions of interest within the image. It serves as a guide for applying seed variations, enabling precise control over where modifications occur. Type should be `IMAGE`.
        - `mask_color` (Required): The specific color in the color mask that identifies the regions of interest. This color acts as a key to isolate parts of the image for seed exploration. Type should be `STRING`.
        - `noise` (Required): The noise pattern to be modified. It represents the base canvas on which seed variations are applied, influencing the final image output. Type should be `NOISE`.
        - `seed_prompt` (Required): A comma-separated list of seeds and their corresponding prompts. These are used to guide the variation process, dictating how and where adjustments are made. Type should be `STRING`.
        - `enable_additional` (Required): A flag indicating whether additional seed and strength parameters should be considered in the variation process. When true, it allows for further customization of the seed exploration. Type should be `BOOLEAN`.
        - `additional_seed` (Required): An additional seed value to be included in the seed exploration process. It offers an extra layer of customization, working alongside the main seed prompts. Type should be `INT`.
        - `additional_strength` (Required): The strength of the additional seed's influence on the variation process. It determines the impact of the additional seed on the final image modifications. Type should be `FLOAT`.
        - `noise_mode` (Required): Specifies the computational device (CPU or GPU) for processing the noise pattern. This choice affects performance and efficiency during the seed exploration. Type should be `COMBO[STRING]`.
        - `variation_method` (Optional): The method used to apply variations to the noise pattern. It defines the algorithmic approach for integrating seed prompts and additional parameters into the image generation process. Type should be `COMBO[STRING]`.
    - Outputs:
        - `noise`: The modified noise pattern after applying seed variations. It reflects the targeted adjustments made based on the input prompts and parameters. Type should be `NOISE`.
        - `mask`: The mask used to define regions of interest within the image, guiding where the seed variations are applied. Type should be `MASK`.