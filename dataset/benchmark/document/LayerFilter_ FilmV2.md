- `LayerFilter_ FilmV2`: The FilmV2 node applies a sophisticated film layer filter to images, enhancing them with a cinematic quality. It leverages advanced image processing techniques to simulate the aesthetic and characteristics of film photography, offering users a unique way to stylize their visuals.
    - Inputs:
        - `image` (Required): The 'image' input type represents the image data to which the FilmV2 filter will be applied. It is crucial for defining the visual content that will undergo the transformation, serving as the foundation for the filter's effects. Type should be `IMAGE`.
        - `center_x` (Required): Specifies the horizontal center of the effect focus area, influencing how certain effects like blur are centered on the image. It plays a key role in the placement of visual enhancements. Type should be `FLOAT`.
        - `center_y` (Required): Defines the vertical center of the effect focus area, affecting the positioning of effects such as blur. Essential for tailoring the filter's impact to specific areas of the image. Type should be `FLOAT`.
        - `saturation` (Required): Controls the intensity of the image's colors. Adjusting saturation can dramatically alter the mood and visual style of the output, making it a key parameter for customization. Type should be `FLOAT`.
        - `vignette_intensity` (Required): Determines the strength of the vignette effect applied to the edges of the image, contributing to the cinematic feel by darkening the periphery. Type should be `FLOAT`.
        - `grain_method` (Required): Selects the algorithm used to add grain to the image, offering options to customize the texture and appearance of the film effect. The choice between 'fastgrain' and 'filmgrainer' methods affects the texture and visual quality of the grain, with each method providing a distinct style and impact on the final image. Type should be `COMBO[STRING]`.
        - `grain_power` (Required): Adjusts the intensity of the grain effect, allowing for fine-tuning the level of perceived film grain and texture. Type should be `FLOAT`.
        - `grain_scale` (Required): Modifies the scale of the grain particles, affecting their size and visibility on the image. This parameter helps in achieving the desired level of detail and texture. Type should be `FLOAT`.
        - `grain_sat` (Required): Sets the saturation level of the grain effect, influencing how colorfully the grain appears on the image. Type should be `FLOAT`.
        - `filmgrainer_shadows` (Required): Controls the visibility of grain in the darker areas of the image, enabling more precise customization of the film effect in shadows. Type should be `FLOAT`.
        - `filmgrainer_highs` (Required): Adjusts the grain effect in the highlights, allowing for nuanced control over the appearance of grain in brighter image areas. Type should be `FLOAT`.
        - `blur_strength` (Required): Determines the intensity of the blur effect, which can be used to simulate depth of field or focus effects. Type should be `INT`.
        - `blur_focus_spread` (Required): Controls the spread of the blur effect around the focus point, affecting how gradually the image transitions from sharp to blurred. Type should be `FLOAT`.
        - `focal_depth` (Required): Sets the depth at which the focus effect is centered, crucial for achieving realistic depth of field simulations. Type should be `FLOAT`.
        - `depth_map` (Optional): An optional input that provides a depth map for more advanced depth-based effects, such as variable blur based on distance from the camera. Including a depth map enables the node to apply blur effects more accurately by simulating the depth of field based on the map, thus enhancing the realism of the film effect. Type should be `IMAGE`.
    - Outputs:
        - `image`: The output 'image' type is the transformed version of the input image, now bearing the distinctive qualities of the FilmV2 filter. It signifies the culmination of the filter's application, showcasing the enhanced cinematic effect. Type should be `IMAGE`.