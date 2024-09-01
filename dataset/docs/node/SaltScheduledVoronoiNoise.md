- `SaltScheduledVoronoiNoise`: The SaltScheduledVoronoiNoise node is designed to generate Voronoi noise-based visual patterns. It allows for the scheduling of various parameters such as scale, detail, and randomness over time, enabling dynamic and evolving visual effects tailored to audio or other time-varying inputs.
    - Parameters:
        - `batch_size`: Specifies the number of patterns to generate in one execution, allowing for batch processing of noise generation. Type should be `INT`.
        - `width`: Determines the width of the generated noise pattern, affecting the spatial resolution. Type should be `INT`.
        - `height`: Sets the height of the generated noise pattern, impacting the vertical resolution. Type should be `INT`.
        - `distance_metric`: Defines the metric used to calculate distances in the Voronoi diagram, influencing the shape and distribution of cells. Type should be `COMBO[STRING]`.
        - `device`: Specifies the computing device (CPU or GPU) where the noise generation process will be executed, affecting performance and capability. Type should be `COMBO[STRING]`.
    - Inputs:
        - `x_schedule`: A schedule for the x-axis positions, enabling dynamic movement or variation of the noise pattern over time. Type should be `LIST`.
        - `y_schedule`: A schedule for the y-axis positions, allowing for vertical movement or variation of the noise pattern over time. Type should be `LIST`.
        - `scale_schedule`: Controls the scale of the noise pattern at different times, enabling zoom in/out effects. Type should be `LIST`.
        - `detail_schedule`: Adjusts the level of detail in the noise pattern over time, affecting the complexity and texture. Type should be `LIST`.
        - `randomness_schedule`: Modifies the randomness of the noise pattern over time, allowing for changes in the pattern's unpredictability. Type should be `LIST`.
        - `seed_schedule`: A schedule for the seed values, enabling the generation of different noise patterns at different times. Type should be `LIST`.
    - Outputs:
        - `images`: The generated Voronoi noise patterns as a batch of images. Type should be `IMAGE`.
        - `batch_size`: The number of generated patterns, confirming the size of the batch processed. Type should be `INT`.