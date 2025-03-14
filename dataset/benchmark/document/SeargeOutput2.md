- `SeargeOutput2`: The SeargeOutput2 node is designed to demultiplex and output a set of generation parameters for image creation, including seed, dimensions, steps, and configuration settings. It plays a crucial role in configuring the generative process by providing detailed control over the generation parameters.
    - Inputs:
        - `parameters` (Required): This parameter contains all the necessary settings for image generation, including seed, image dimensions, steps, CFG scale, sampler and scheduler names, and options for saving images. It's essential for defining the behavior and output of the generative process. Type should be `PARAMETERS`.
    - Outputs:
        - `parameters`: Returns the original set of parameters provided as input, facilitating further processing or utilization in the workflow. Type should be `PARAMETERS`.
        - `seed`: The seed value for random number generation, ensuring reproducibility of the generated images. Type should be `INT`.
        - `image_width`: The width of the generated image in pixels. Type should be `INT`.
        - `image_height`: The height of the generated image in pixels. Type should be `INT`.
        - `steps`: The number of steps to run the generation process, affecting the detail and quality of the output. Type should be `INT`.
        - `cfg`: The CFG scale used to control the trade-off between fidelity to the text prompt and the randomness of the generated images. Type should be `FLOAT`.
        - `sampler_name`: The name of the sampler algorithm used for image generation. Type should be `SAMPLER_NAME`.
        - `scheduler`: The name of the scheduler algorithm used to manage the generation steps. Type should be `SCHEDULER_NAME`.
        - `save_image`: A boolean indicating whether the generated image should be saved. Type should be `ENABLE_STATE`.
        - `save_directory`: The directory path where the generated image will be saved, if applicable. Type should be `SAVE_FOLDER`.
