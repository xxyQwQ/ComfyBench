- `LayerStyle_ InnerGlow V2`: The InnerGlow V2 node is designed to apply an inner glow effect to images, enhancing their visual appeal by adding a soft light from within. This node focuses on creating a more vibrant and dynamic look for graphical elements by simulating the effect of light emanating from the inside out.
    - Inputs:
        - `background_image` (Required): The base image over which the inner glow effect will be applied, serving as the backdrop for the effect. Type should be `IMAGE`.
        - `layer_image` (Required): The image layer to which the inner glow effect will be applied, directly influencing the appearance of the glow. Type should be `IMAGE`.
        - `invert_mask` (Required): A boolean flag to invert the mask applied to the layer image, affecting the areas where the glow effect is visible. Type should be `BOOLEAN`.
        - `blend_mode` (Required): Specifies the blending mode used to combine the glow effect with the layer image, impacting the overall look of the effect. Type should be `COMBO[STRING]`.
        - `opacity` (Required): Controls the opacity level of the inner glow effect, allowing for fine-tuning of its visibility. Type should be `INT`.
        - `brightness` (Required): Adjusts the brightness of the inner glow effect, influencing its intensity and spread. Type should be `INT`.
        - `glow_range` (Required): Defines the range of the glow effect, determining how far the glow extends from the layer image. Type should be `INT`.
        - `blur` (Required): Sets the blur amount for the glow effect, contributing to its softness and diffusion. Type should be `INT`.
        - `light_color` (Required): The color at the center of the glow effect, setting the tone for the light source. Type should be `STRING`.
        - `glow_color` (Required): The color at the outer edge of the glow effect, defining the peripheral color transition. Type should be `STRING`.
        - `layer_mask` (Optional): An optional mask that can be applied to the layer image, dictating the specific areas where the glow effect is applied. Type should be `MASK`.
    - Outputs:
        - `image`: The resulting image after applying the inner glow effect, showcasing the enhanced visual appeal with a soft, emanating light. Type should be `IMAGE`.
