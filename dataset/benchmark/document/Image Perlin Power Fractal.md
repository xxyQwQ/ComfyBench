- `Image Perlin Power Fractal`: This node generates a Perlin Power Fractal image, leveraging Perlin noise to create complex, visually appealing textures or landscapes. It allows for detailed customization of the fractal's appearance through various parameters, including scale, octaves, and persistence.
    - Inputs:
        - `width` (Required): Specifies the width of the generated image. Affects the overall size and detail level of the fractal. Type should be `INT`.
        - `height` (Required): Determines the height of the generated image, influencing the fractal's size and detail. Type should be `INT`.
        - `scale` (Required): Controls the scale of the noise used in the fractal, impacting the zoom level and detail of the texture. Type should be `INT`.
        - `octaves` (Required): Adjusts the number of layers of noise to combine, affecting the complexity and detail of the fractal. Type should be `INT`.
        - `persistence` (Required): Modifies the amplitude of each octave, influencing the roughness and detail of the fractal's surface. Type should be `FLOAT`.
        - `lacunarity` (Required): Changes the frequency of each octave, affecting the fractal's texture and detail. Type should be `FLOAT`.
        - `exponent` (Required): Alters the power to which the scale is raised, impacting the fractal's appearance and detail. Type should be `FLOAT`.
        - `seed` (Required): Sets the seed for the noise generation, ensuring reproducibility of the fractal patterns. Type should be `INT`.
    - Outputs:
        - `image`: The generated Perlin Power Fractal image, represented as a tensor. Type should be `IMAGE`.
