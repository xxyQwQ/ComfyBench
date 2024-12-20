- `RelightSimple`: The RelightSimple node is designed to adjust the lighting of an image based on provided normal maps and directional light parameters. It allows for dynamic relighting effects, simulating different lighting conditions by altering the direction and brightness of the light source.
    - Inputs:
        - `image` (Required): The input image to be relit. It serves as the base for the lighting adjustments. Type should be `IMAGE`.
        - `normals` (Required): Normal maps corresponding to the input image, used to calculate how light interacts with the surface. Type should be `IMAGE`.
        - `x_dir` (Required): The x-direction component of the light source, influencing the direction from which the light appears to come. Type should be `FLOAT`.
        - `y_dir` (Required): The y-direction component of the light source, affecting the vertical angle of the incoming light. Type should be `FLOAT`.
        - `brightness` (Required): Controls the intensity of the light source, allowing for brighter or dimmer lighting effects. Type should be `FLOAT`.
    - Outputs:
        - `image`: The output image with adjusted lighting based on the provided normals and light direction parameters. Type should be `IMAGE`.
