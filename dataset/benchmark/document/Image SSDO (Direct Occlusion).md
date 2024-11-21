- `Image SSDO (Direct Occlusion)`: This node is designed to simulate the effect of direct occlusion in images, enhancing the realism of 3D scenes by calculating how objects block light sources. It processes a set of images along with their depth information to generate occlusion effects, which can be adjusted for strength, radius, and specular highlights, optionally incorporating colored occlusion for added visual depth.
    - Inputs:
        - `images` (Required): A collection of images on which to perform direct occlusion. These images serve as the base for calculating how light interacts with objects, directly affecting the visual outcome of the occlusion effect. Type should be `IMAGE`.
        - `depth_images` (Required): Corresponding depth information for each image, used to determine the spatial relationships between objects in the scene. This depth data is crucial for accurately simulating how objects block or allow light to pass, influencing the occlusion effect. Type should be `IMAGE`.
        - `strength` (Required): Controls the intensity of the occlusion effect, allowing for fine-tuning of how pronounced the occlusion appears in the final image. Type should be `FLOAT`.
        - `radius` (Required): Determines the size of the area around objects affected by occlusion, impacting the softness and spread of shadows. Type should be `FLOAT`.
        - `specular_threshold` (Required): Sets the threshold for specular highlights, enabling the simulation of shiny surfaces where light is more likely to reflect. Type should be `INT`.
        - `colored_occlusion` (Required): When enabled, applies color to the occlusion effect, adding a layer of visual complexity and realism to the scene. Type should be `COMBO[STRING]`.
    - Outputs:
        - `composited_images`: The final images with direct occlusion applied, showcasing enhanced depth and realism through the simulation of light blocking effects. Type should be `IMAGE`.
        - `ssdo_images`: The calculated occlusion effects for each image, which can be used for further analysis or processing. Type should be `IMAGE`.
        - `ssdo_image_masks`: Masks indicating the areas of each image affected by occlusion, useful for understanding the spatial distribution of the effect. Type should be `IMAGE`.
        - `light_source_image_masks`: Identified light sources in the images, which are crucial for accurately simulating the occlusion effect. Type should be `IMAGE`.