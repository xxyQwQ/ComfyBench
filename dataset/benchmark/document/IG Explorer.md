- `IG Explorer`: The IG Explorer node is designed for navigating and analyzing latent space, facilitating the exploration and manipulation of parameters to generate or modify content.
    - Inputs:
        - `folder` (Required): Specifies the directory containing the job queue and other relevant files for exploration, acting as the starting point for the node's operations. Type should be `STRING`.
        - `precision` (Required): Determines the level of detail or accuracy in the exploration process, affecting how finely the node examines the latent space. Type should be `FLOAT`.
    - Outputs:
        - `float`: Outputs a float value determined through exploration, indicating a specific point or configuration in latent space. Type should be `FLOAT`.
        - `string`: Provides the filename of the image generated or identified during the exploration process, reflecting the current state of exploration. Type should be `STRING`.
