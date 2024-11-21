- `LayerUtility_ ImageBlend V2`: This node facilitates the blending of two images, offering a range of blending modes and control over the blend's opacity and mask inversion. It's designed to enhance creative workflows in image processing by allowing for complex visual effects through layer blending.
    - Inputs:
        - `background_image` (Required): Serves as the base layer for blending, providing the foundational imagery over which the layer image is blended. Type should be `IMAGE`.
        - `layer_image` (Required): Acts as the top layer in the blend, overlaying the background image according to the specified blending mode and opacity. Type should be `IMAGE`.
        - `invert_mask` (Required): Determines whether the mask applied to the layer image should be inverted, affecting how the blend is masked out. Type should be `BOOLEAN`.
        - `blend_mode` (Required): Specifies the method of blending the layer image with the background, such as multiply or overlay, influencing the visual outcome. Type should be `COMBO[STRING]`.
        - `opacity` (Required): Controls the transparency level of the layer image over the background, allowing for fine-tuning of the blend intensity. Type should be `INT`.
        - `layer_mask` (Optional): Optional mask that can be applied to the layer image, defining areas of transparency for intricate blending effects. Type should be `MASK`.
    - Outputs:
        - `image`: The result of blending the layer image with the background image, reflecting the chosen blend mode, opacity, and mask settings. Type should be `IMAGE`.