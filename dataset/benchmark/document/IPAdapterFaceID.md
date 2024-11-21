- `IPAdapterFaceID`: The IPAdapterFaceID node is designed to enhance image processing capabilities by integrating face identification features. It leverages advanced IPAdapter functionalities to apply face-specific adjustments and embeddings, aiming to improve the quality and relevance of generated images based on facial recognition.
    - Inputs:
        - `model` (Required): Specifies the model to be used for face identification, central to the node's operation. Type should be `MODEL`.
        - `ipadapter` (Required): Defines the IPAdapter configuration to be used, crucial for determining how face identification features are integrated. Type should be `IPADAPTER`.
        - `image` (Required): The input image to be processed, serving as the basis for face identification and subsequent adjustments. Type should be `IMAGE`.
        - `weight` (Required): A float value that adjusts the influence of the face identification features on the final image output. Type should be `FLOAT`.
        - `weight_faceidv2` (Required): A float value specifically for adjusting the influence of FaceID v2 features, offering finer control over the face identification process. Type should be `FLOAT`.
        - `weight_type` (Required): Determines the method of weighting face identification features, affecting how these features influence the final image. Type should be `COMBO[STRING]`.
        - `combine_embeds` (Required): Specifies the method for combining face identification embeddings, impacting the integration of facial features into the image. Type should be `COMBO[STRING]`.
        - `start_at` (Required): A float value indicating the starting point for applying face identification features within the processing pipeline. Type should be `FLOAT`.
        - `end_at` (Required): A float value indicating the end point for applying face identification features, defining the scope of their influence. Type should be `FLOAT`.
        - `embeds_scaling` (Required): Describes how face identification embeddings are scaled, affecting their impact on the image processing. Type should be `COMBO[STRING]`.
        - `image_negative` (Optional): An optional input image to be used in a negative context, providing additional control over the face identification process. Type should be `IMAGE`.
        - `attn_mask` (Optional): An optional mask to focus or limit the attention mechanism during face identification, enhancing processing accuracy. Type should be `MASK`.
        - `clip_vision` (Optional): An optional CLIP vision model to further refine the face identification process through visual context. Type should be `CLIP_VISION`.
        - `insightface` (Optional): An optional InsightFace model required for certain face identification functionalities, enhancing the node's capabilities. Type should be `INSIGHTFACE`.
    - Outputs:
        - `MODEL`: The processed model with integrated face identification features, ready for further image generation tasks. Type should be `MODEL`.
        - `face_image`: The output image that has been enhanced with face identification features, showcasing improved facial recognition and integration. Type should be `IMAGE`.