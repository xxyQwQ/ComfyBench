- `SamplerLCMUpscale`: The SamplerLCMUpscale node is designed to upscale images using various methods, providing a flexible approach to image sampling with adjustable scale ratios and steps. It leverages a custom sampling function to enhance image resolution, catering to different upscale needs and preferences.
    - Inputs:
        - `scale_ratio` (Required): Specifies the ratio by which the image should be upscaled. A higher value results in a larger image. Type should be `FLOAT`.
        - `scale_steps` (Required): Determines the number of steps to perform the upscale process. A value of -1 indicates automatic determination based on the scale ratio. Type should be `INT`.
        - `upscale_method` (Required): Selects the method used for upscaling the image, offering various algorithms to suit different quality and performance requirements. Type should be `COMBO[STRING]`.
    - Outputs:
        - `sampler`: Produces a sampler configured to upscale images according to the specified parameters. Type should be `SAMPLER`.