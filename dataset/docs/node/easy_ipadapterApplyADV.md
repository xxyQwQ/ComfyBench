- `easy ipadapterApplyADV`: This node specializes in applying advanced IPAdapter configurations to models, enhancing their capabilities with custom image processing and adaptation strategies. It leverages IPAdapter to modify and fine-tune model behaviors based on specific image attributes, presets, and adaptation parameters, aiming to achieve optimal integration and performance for specialized tasks.
    - Parameters:
        - `preset`: A predefined configuration or set of parameters that dictate how the IPAdapter should modify the model, tailoring its behavior to specific needs. Type should be `COMBO[STRING]`.
        - `lora_strength`: Specifies the strength of the LoRA model adjustment. Type should be `FLOAT`.
        - `provider`: Defines the computation provider for the operation, such as CPU or GPU. Type should be `COMBO[STRING]`.
        - `weight`: The weight factor for the adaptation process, influencing how the IPAdapter modifies the model. Type should be `FLOAT`.
        - `weight_faceidv2`: Specific weight adjustment for FaceID v2 features within the adaptation process. Type should be `FLOAT`.
        - `weight_type`: Specifies the type of weighting method used in the adaptation process, such as linear or non-linear. Type should be `COMBO[STRING]`.
        - `combine_embeds`: Describes how multiple embeddings are combined during the adaptation process, e.g., concatenation or averaging. Type should be `COMBO[STRING]`.
        - `start_at`: Defines the starting point of the adaptation effect on the model. Type should be `FLOAT`.
        - `end_at`: Specifies the end point of the adaptation effect on the model, allowing for precise control over the adaptation range. Type should be `FLOAT`.
        - `embeds_scaling`: Defines how embeddings are scaled or adjusted during the adaptation, influencing the final output. Type should be `COMBO[STRING]`.
        - `cache_mode`: Determines the caching strategy for the adaptation process, affecting performance and resource utilization. Type should be `COMBO[STRING]`.
        - `use_tiled`: Indicates whether a tiled adaptation approach should be used, affecting the processing of large images. Type should be `BOOLEAN`.
        - `use_batch`: Indicates whether batch processing is utilized for efficiency in handling multiple images or data points. Type should be `BOOLEAN`.
        - `sharpening`: Specifies the level of sharpening applied to the adapted images, enhancing clarity and detail. Type should be `FLOAT`.
    - Inputs:
        - `model`: The model to which the IPAdapter will be applied, serving as the base for further adaptations. Type should be `MODEL`.
        - `image`: The image to be processed or adapted by the IPAdapter, acting as a key input for the adaptation process. Type should be `IMAGE`.
        - `image_negative`: An optional negative image input for the adaptation process, used to negate certain features or effects. Type should be `IMAGE`.
        - `attn_mask`: An optional attention mask to focus or ignore specific parts of the image during adaptation. Type should be `MASK`.
        - `clip_vision`: Optional CLIP vision model input to guide the adaptation process based on visual concepts. Type should be `CLIP_VISION`.
        - `optional_ipadapter`: An optional IPAdapter instance to be used in conjunction with the primary adaptation process. Type should be `IPADAPTER`.
    - Outputs:
        - `model`: The modified model after applying the IPAdapter, showcasing enhanced or altered capabilities. Type should be `MODEL`.
        - `images`: The images resulting from the adaptation process, potentially modified or enhanced. Type should be `IMAGE`.
        - `masks`: The masks generated during the adaptation process, used for focusing or excluding specific image areas. Type should be `MASK`.
        - `ipadapter`: The IPAdapter instance used for the adaptation, encapsulating the specific configurations applied. Type should be `IPADAPTER`.