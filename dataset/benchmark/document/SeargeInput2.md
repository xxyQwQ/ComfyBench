- `SeargeInput2`: The node is designed to facilitate the integration and manipulation of various input types for generating or modifying content, focusing on enhancing user interaction with the system by providing a structured way to input data.
    - Inputs:
        - `seed` (Required): Specifies the initial value for random number generation, ensuring reproducibility and consistency in the generated content. Type should be `INT`.
        - `image_width` (Required): Determines the width of the generated image, allowing for customization of the output's dimensions. Type should be `INT`.
        - `image_height` (Required): Sets the height of the generated image, enabling control over the output's size. Type should be `INT`.
        - `steps` (Required): Defines the number of steps the generation process should take, affecting the detail and quality of the output. Type should be `INT`.
        - `cfg` (Required): Controls the configuration setting for the generation process, influencing the output's characteristics. Type should be `FLOAT`.
        - `sampler_name` (Required): Selects the sampling algorithm used during generation, impacting the diversity and quality of results. Type should be `COMBO[STRING]`.
        - `scheduler` (Required): Chooses the scheduling algorithm for the generation process, affecting the progression and quality of outputs. Type should be `COMBO[STRING]`.
        - `save_image` (Required): Determines whether the generated image should be saved, facilitating content preservation. Type should be `COMBO[STRING]`.
        - `save_directory` (Required): Specifies the directory where generated images will be saved, organizing output management. Type should be `COMBO[STRING]`.
        - `inputs` (Optional): Optional additional inputs that can be provided for the generation process, offering further customization. Type should be `PARAMETER_INPUTS`.
    - Outputs:
        - `inputs`: Outputs the parameters used in the generation process, encapsulating the configured inputs for potential reuse or analysis. Type should be `PARAMETER_INPUTS`.