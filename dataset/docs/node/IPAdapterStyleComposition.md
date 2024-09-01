- `IPAdapterStyleComposition`: This node specializes in the application of style and composition adjustments to images using IP adapters, enabling the enhancement or alteration of visual elements through advanced image processing techniques.
    - Parameters:
        - `weight_style`: Determines the intensity of the style adjustments, affecting the degree to which the style influences the output. Type should be `FLOAT`.
        - `weight_composition`: Controls the intensity of the composition adjustments, impacting how significantly the composition alters the output. Type should be `FLOAT`.
        - `expand_style`: A boolean flag that, when true, expands the influence of style adjustments beyond their typical scope, potentially leading to more pronounced stylistic changes. Type should be `BOOLEAN`.
        - `combine_embeds`: Specifies the method for combining embeddings from different sources, affecting how style and composition adjustments are integrated. Type should be `COMBO[STRING]`.
        - `start_at`: Defines the starting point in the process for applying adjustments, allowing for precise control over when the style and composition changes begin to take effect. Type should be `FLOAT`.
        - `end_at`: Sets the endpoint in the process for applying adjustments, determining when the style and composition changes cease to influence the output. Type should be `FLOAT`.
        - `embeds_scaling`: Dictates the approach to scaling embeddings, influencing how adjustments are calibrated and applied to the images. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: Specifies the model to be used for the style and composition adjustments, serving as the foundation for the image processing operations. Type should be `MODEL`.
        - `ipadapter`: Defines the IP adapter to be utilized for applying the style and composition adjustments to the images. Type should be `IPADAPTER`.
        - `image_style`: The image to which the style adjustments will be applied, dictating the aesthetic direction of the output. Type should be `IMAGE`.
        - `image_composition`: The image to which the composition adjustments will be applied, influencing the structural aspects of the output. Type should be `IMAGE`.
        - `image_negative`: An optional image that represents undesired elements or styles, guiding the adjustments away from these characteristics. Type should be `IMAGE`.
        - `attn_mask`: An optional mask that can be applied to focus or restrict adjustments to specific areas of the image, enhancing precision. Type should be `MASK`.
        - `clip_vision`: An optional parameter that, when provided, utilizes CLIP vision models to further refine the style and composition adjustments based on semantic understanding. Type should be `CLIP_VISION`.
    - Outputs:
        - `model`: The modified model after applying style and composition adjustments, reflecting the changes made to the visual elements. Type should be `MODEL`.