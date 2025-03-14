- `Sample Image Area as Palette [Dream]`: This node samples colors from a specified area within an image to create a color palette. It allows for targeted palette generation by focusing on specific regions of an image, enhancing thematic consistency and relevance in the resulting palette.
    - Inputs:
        - `image` (Required): The image from which colors are sampled to create the palette. This parameter is crucial for defining the source of the color extraction. Type should be `IMAGE`.
        - `samples` (Required): Specifies the number of color samples to extract from the designated area of the image, directly influencing the diversity and richness of the resulting palette. Type should be `INT`.
        - `seed` (Required): A seed for the random number generator, ensuring reproducibility of the palette by controlling the randomness in color sampling. Type should be `INT`.
        - `area` (Required): Defines the specific area of the image from which to sample colors, allowing for targeted palette creation based on predefined image regions. Type should be `COMBO[STRING]`.
    - Outputs:
        - `palette`: The generated color palette, comprising colors sampled from the specified area of the image. Type should be `RGB_PALETTE`.
