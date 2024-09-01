- `IPAdapterFromParams`: The IPAdapterFromParams node is designed to dynamically configure and apply image processing adapters based on a set of parameters. It allows for the customization of image processing techniques by combining various embedding strategies and scaling methods, facilitating advanced image manipulation and enhancement.
    - Parameters:
        - `combine_embeds`: Determines how multiple embeddings are combined, affecting the final image output through various strategies like concatenation or averaging. Type should be `COMBO[STRING]`.
        - `embeds_scaling`: Controls the scaling of embeddings, influencing the adaptation's impact on the image based on selected methods. Type should be `COMBO[STRING]`.
    - Inputs:
        - `model`: Specifies the model to which the image processing adapter will be applied, serving as the foundation for the adaptation process. Type should be `MODEL`.
        - `ipadapter`: Identifies the specific image processing adapter to be used, dictating the core processing technique. Type should be `IPADAPTER`.
        - `ipadapter_params`: Contains parameters that fine-tune the image processing adapter's behavior, offering detailed control over the adaptation process. Type should be `IPADAPTER_PARAMS`.
        - `image_negative`: Optionally provides a negative image input, used to inversely influence the adaptation process. Type should be `IMAGE`.
        - `clip_vision`: Optionally integrates CLIP vision embeddings, enhancing the adaptation with semantic understanding from textual descriptions. Type should be `CLIP_VISION`.
    - Outputs:
        - `model`: Outputs the adapted model, reflecting the applied image processing techniques and parameter adjustments. Type should be `MODEL`.