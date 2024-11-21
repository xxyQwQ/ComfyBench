- `IPAdapterRegionalConditioning`: This node specializes in applying regional conditioning to images within the IPAdapter framework. It leverages masks to selectively apply positive and negative prompts to specific areas of an image, enabling fine-grained control over the generation process. This functionality is crucial for tasks requiring targeted adjustments or enhancements, such as modifying specific image regions without affecting the overall composition.
    - Inputs:
        - `image` (Required): The image to be conditioned. It serves as the base for applying regional conditioning, determining the areas where positive or negative prompts will be applied. Type should be `IMAGE`.
        - `image_weight` (Required): A weight parameter for the image, influencing the degree to which the original image is retained or modified during the conditioning process. Type should be `FLOAT`.
        - `prompt_weight` (Required): The weight assigned to the prompts, dictating the intensity of their influence on the conditioned regions of the image. Type should be `FLOAT`.
        - `weight_type` (Required): Specifies the type of weighting mechanism used for conditioning, affecting how image and prompt weights are balanced. Type should be `COMBO[STRING]`.
        - `start_at` (Required): Defines the starting layer for applying conditioning, allowing for layer-specific adjustments. Type should be `FLOAT`.
        - `end_at` (Required): Determines the ending layer for conditioning, marking the depth of influence through the network layers. Type should be `FLOAT`.
        - `mask` (Optional): An optional mask to define specific regions for conditioning, enabling targeted application of positive or negative prompts. Type should be `MASK`.
        - `positive` (Optional): Optional positive prompts to be applied to the masked regions, enhancing or adding desired features. Type should be `CONDITIONING`.
        - `negative` (Optional): Optional negative prompts to be applied, used to diminish or remove undesired aspects within the masked areas. Type should be `CONDITIONING`.
    - Outputs:
        - `IPADAPTER_PARAMS`: The configured parameters for the IPAdapter, ready for integration with the image processing pipeline. Type should be `IPADAPTER_PARAMS`.
        - `POSITIVE`: The processed positive prompts, adjusted according to the conditioning settings. Type should be `CONDITIONING`.
        - `NEGATIVE`: The processed negative prompts, tailored to the specified conditioning requirements. Type should be `CONDITIONING`.