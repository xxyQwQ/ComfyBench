- `ADE_CustomCFG`: This node is designed for creating custom configuration settings for animation diffusion processes, allowing users to specify multiple values for configuration settings that influence the generation process.
    - Inputs:
        - `cfg_multival` (Required): Specifies the multiple values for configuration settings, enabling fine-tuned control over the animation diffusion process. Type should be `MULTIVAL`.
    - Outputs:
        - `custom_cfg`: Outputs a custom configuration group, encapsulating the specified configuration settings for use in animation diffusion. Type should be `CUSTOM_CFG`.
