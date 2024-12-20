- `BasicScheduler`: The BasicScheduler node is designed to calculate and adjust the sigma values for a given model and scheduler over a specified number of steps, incorporating an optional denoise parameter to refine the process. It serves as a foundational element in custom sampling strategies, enabling precise control over the diffusion process in generative models.
    - Inputs:
        - `model` (Required): Specifies the model for which sigma values are to be calculated, playing a crucial role in determining the diffusion process's behavior. Type should be `MODEL`.
        - `scheduler` (Required): Defines the scheduler to be used for calculating sigma values, directly influencing the diffusion steps and their granularity. Type should be `COMBO[STRING]`.
        - `steps` (Required): Determines the number of diffusion steps to be used, affecting the resolution and granularity of the sigma values. Type should be `INT`.
        - `denoise` (Required): Optional parameter to adjust the effective number of steps based on denoising level, allowing for finer control over the diffusion process. Type should be `FLOAT`.
    - Outputs:
        - `sigmas`: The calculated sigma values for the specified model and scheduler, essential for controlling the diffusion process. Type should be `SIGMAS`.
