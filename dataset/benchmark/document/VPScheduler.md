- `VPScheduler`: The VPScheduler node is designed to generate a sequence of noise levels (sigmas) for variational path sampling in diffusion models, based on specified scheduling parameters.
    - Inputs:
        - `steps` (Required): Defines the total number of steps in the noise schedule, affecting the granularity and length of the generated sigma sequence. Type should be `INT`.
        - `beta_d` (Required): Specifies the beta decay parameter, influencing the rate at which noise levels decrease throughout the schedule. Type should be `FLOAT`.
        - `beta_min` (Required): Sets the minimum beta value, determining the lowest noise level in the schedule. Type should be `FLOAT`.
        - `eps_s` (Required): Adjusts the epsilon start value, fine-tuning the initial noise level. Type should be `FLOAT`.
    - Outputs:
        - `sigmas`: A sequence of noise levels (sigmas) calculated for variational path sampling, tailored to the input scheduling parameters. Type should be `SIGMAS`.
