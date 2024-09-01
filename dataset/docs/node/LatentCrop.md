- `LatentCrop`: The LatentCrop node is designed to perform cropping operations on latent representations of images. It allows for the specification of the crop dimensions and position, enabling targeted modifications of the latent space.
    - Parameters:
        - `width`: Specifies the width of the crop area. It directly influences the dimensions of the output latent representation. Type should be `INT`.
        - `height`: Specifies the height of the crop area, affecting the size of the resulting cropped latent representation. Type should be `INT`.
        - `x`: Determines the starting x-coordinate of the crop area, influencing the position of the crop within the original latent representation. Type should be `INT`.
        - `y`: Determines the starting y-coordinate of the crop area, setting the position of the crop within the original latent representation. Type should be `INT`.
    - Inputs:
        - `samples`: The 'samples' parameter represents the latent representations to be cropped. It is crucial for defining the data on which the cropping operation will be performed. Type should be `LATENT`.
    - Outputs:
        - `latent`: The output is a modified latent representation with the specified crop applied. Type should be `LATENT`.