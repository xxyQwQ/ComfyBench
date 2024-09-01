- `BetterFilmGrain`: The BetterFilmGrain node enhances images by applying a customizable film grain effect, simulating the texture and appearance of traditional film photography. It allows for fine-tuning of the grain's scale, strength, saturation, and overall tone to achieve a desired aesthetic.
    - Parameters:
        - `scale`: Determines the scale of the grain particles, affecting their size relative to the image. A smaller scale results in finer grain. Type should be `FLOAT`.
        - `strength`: Controls the intensity of the grain effect, with higher values making the grain more pronounced. Type should be `FLOAT`.
        - `saturation`: Adjusts the color saturation of the grain effect, allowing for more or less colorful grain textures. Type should be `FLOAT`.
        - `toe`: Modifies the toe of the film response curve, affecting the shadow tones and overall contrast of the grain effect. Type should be `FLOAT`.
        - `seed`: A seed value for random number generation, ensuring reproducibility of the grain pattern. Type should be `INT`.
    - Inputs:
        - `image`: The input image to which the film grain effect will be applied. It serves as the base for the grain texture overlay. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output image with the applied film grain effect, showcasing enhanced texture and a film-like aesthetic. Type should be `IMAGE`.