- `easy ipadapterApplyEmbeds`: The node 'easy ipadapterApplyEmbeds' is designed to apply embedding transformations to models using IPAdapter, facilitating the integration of positional and negative embeddings into the model's processing pipeline. It abstracts the complexity of embedding manipulation, offering a streamlined approach to enhance model performance with custom embeddings.
    - Inputs:
        - `model` (Required): Specifies the model to which embeddings will be applied, serving as the foundational structure for embedding integration. Type should be `MODEL`.
        - `clip_vision` (Required): Indicates whether to apply CLIP vision embeddings, influencing the model's visual understanding and processing. Type should be `CLIP_VISION`.
        - `ipadapter` (Required): The IPAdapter instance used for embedding transformations, central to the embedding application process. Type should be `IPADAPTER`.
        - `pos_embed` (Required): The positive embeddings to be applied, enhancing the model's positive feature recognition. Type should be `EMBEDS`.
        - `weight` (Required): Defines the weight of the embeddings, influencing their impact on the model. Type should be `FLOAT`.
        - `weight_type` (Required): Specifies the type of weight applied to the embeddings, affecting their integration into the model. Type should be `COMBO[STRING]`.
        - `start_at` (Required): Determines the starting point for embedding application, allowing for precise control over when embeddings influence the model. Type should be `FLOAT`.
        - `end_at` (Required): Sets the endpoint for embedding application, defining the scope of embedding influence on the model. Type should be `FLOAT`.
        - `embeds_scaling` (Required): Defines the scaling method for embeddings, affecting how embeddings are integrated and weighted within the model. Type should be `COMBO[STRING]`.
        - `neg_embed` (Optional): The negative embeddings to be applied, enhancing the model's negative feature recognition. Type should be `EMBEDS`.
        - `attn_mask` (Optional): An optional attention mask for more targeted embedding application, providing additional control over how embeddings affect the model. Type should be `MASK`.
    - Outputs:
        - `model`: The enhanced model with applied embeddings, reflecting the integration of positive and negative embeddings. Type should be `MODEL`.
        - `ipadapter`: The IPAdapter instance after embedding application, indicating the successful integration of embeddings. Type should be `IPADAPTER`.