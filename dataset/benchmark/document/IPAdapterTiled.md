- `IPAdapterTiled`: The IPAdapterTiled node is designed to apply image processing adaptations in a tiled manner, allowing for detailed and scalable modifications to images. It leverages various image processing techniques and parameters to enhance, modify, or transform images based on the provided inputs and configurations.
    - Inputs:
        - `model` (Required): Specifies the model to be used for image processing, serving as the core component for adaptations. Type should be `MODEL`.
        - `ipadapter` (Required): Defines the IPAdapter configuration to be applied, dictating the specific image processing techniques and parameters. Type should be `IPADAPTER`.
        - `image` (Required): The input image to be processed, serving as the primary subject for the adaptations. Type should be `IMAGE`.
        - `weight` (Required): A floating-point value that adjusts the intensity of the applied adaptations, allowing for fine-tuned control over the output. Type should be `FLOAT`.
        - `weight_type` (Required): Determines the method of weighting the adaptations, influencing how different aspects of the image are emphasized or blended. Type should be `COMBO[STRING]`.
        - `combine_embeds` (Required): Determines how embeddings are combined, offering options like concatenation to influence the final image adaptation. Type should be `COMBO[STRING]`.
        - `start_at` (Required): A floating-point value indicating the starting point of the adaptations, enabling phased or gradual application. Type should be `FLOAT`.
        - `end_at` (Required): A floating-point value marking the end point of the adaptations, allowing for precise control over the extent of modifications. Type should be `FLOAT`.
        - `sharpening` (Required): A floating-point value that controls the level of sharpening applied to the image, enhancing detail and clarity. Type should be `FLOAT`.
        - `embeds_scaling` (Required): Specifies the scaling method for embeddings, affecting the adaptation's influence on different image features. Type should be `COMBO[STRING]`.
        - `image_negative` (Optional): An optional input image representing negative aspects to be minimized or avoided in the adaptations. Type should be `IMAGE`.
        - `attn_mask` (Optional): An optional mask that directs attention to specific areas of the image, focusing adaptations where needed. Type should be `MASK`.
        - `clip_vision` (Optional): An optional parameter that integrates CLIP vision models for guided image adaptations, enhancing relevance and coherence. Type should be `CLIP_VISION`.
    - Outputs:
        - `MODEL`: The processed model after image adaptations have been applied. Type should be `MODEL`.
        - `tiles`: The resulting image tiles after processing, showcasing the segmented adaptations. Type should be `IMAGE`.
        - `masks`: The masks applied to each tile, indicating areas of focus or modification. Type should be `MASK`.