- `ttN pipeLoader_v2`: The `ttN pipeLoader_v2` node is designed for advanced data loading and preprocessing within a pipeline. It focuses on efficiently handling and transforming data to prepare it for subsequent processing or analysis steps, leveraging enhanced capabilities over its predecessor for improved performance and flexibility.
    - Inputs:
        - `ckpt_name` (Required): Specifies the checkpoint name for model loading, affecting the pipeline's configuration. Type should be `COMBO[STRING]`.
        - `config_name` (Required): Specifies the configuration name for the pipeline, affecting how data is processed and managed. Type should be `COMBO[STRING]`.
        - `vae_name` (Required): Specifies the VAE model name for loading, affecting the pipeline's configuration and capabilities. Type should be `COMBO[STRING]`.
        - `clip_skip` (Required): Determines the number of CLIP layers to skip, affecting the data preprocessing phase. Type should be `INT`.
        - `loras` (Required): Specifies the LoRA models to be used, affecting the pipeline's performance and output. Type should be `STRING`.
        - `positive` (Required): unknown Type should be `STRING`.
        - `positive_token_normalization` (Required): Indicates the method for normalizing positive tokens, affecting text processing. Type should be `COMBO[STRING]`.
        - `positive_weight_interpretation` (Required): Specifies how the weights for positive tokens are interpreted, affecting text processing. Type should be `COMBO[STRING]`.
        - `negative` (Required): unknown Type should be `STRING`.
        - `negative_token_normalization` (Required): Indicates the method for normalizing negative tokens, affecting text processing. Type should be `COMBO[STRING]`.
        - `negative_weight_interpretation` (Required): Specifies how the weights for negative tokens are interpreted, affecting text processing. Type should be `COMBO[STRING]`.
        - `empty_latent_aspect` (Required): Specifies the aspect ratio for empty latent images, affecting the pipeline's image processing capabilities. Type should be `COMBO[STRING]`.
        - `empty_latent_width` (Required): unknown Type should be `INT`.
        - `empty_latent_height` (Required): unknown Type should be `INT`.
        - `batch_size` (Required): Specifies the batch size for processing, affecting the pipeline's efficiency and throughput. Type should be `INT`.
        - `seed` (Required): Specifies the seed for random number generation in the pipeline, ensuring reproducibility. Type should be `INT`.
        - `model_override` (Optional): Allows for overriding the default model, affecting the pipeline's flexibility and output. Type should be `MODEL`.
        - `clip_override` (Optional): Allows for overriding the default CLIP model, affecting the pipeline's flexibility and output. Type should be `CLIP`.
        - `optional_lora_stack` (Optional): Specifies an optional stack of LoRA models, enhancing the pipeline's adaptability and performance. Type should be `LORA_STACK`.
        - `optional_controlnet_stack` (Optional): Specifies an optional stack of ControlNet models, enhancing the pipeline's adaptability and performance. Type should be `CONTROL_NET_STACK`.
        - `prepend_positive` (Optional): Specifies text to prepend to positive conditioning, affecting text processing. Type should be `STRING`.
        - `prepend_negative` (Optional): Specifies text to prepend to negative conditioning, affecting text processing. Type should be `STRING`.
    - Outputs:
        - `pipe`: Outputs a modified pipeline object, incorporating the loaded and preprocessed data ready for further stages. Type should be `PIPE_LINE`.
        - `model`: Outputs the model used in the pipeline after processing. Type should be `MODEL`.
        - `positive`: Outputs the positive conditioning used in the pipeline after processing. Type should be `CONDITIONING`.
        - `negative`: Outputs the negative conditioning used in the pipeline after processing. Type should be `CONDITIONING`.
        - `latent`: Outputs the latent variables used in the pipeline after processing. Type should be `LATENT`.
        - `vae`: Outputs the VAE model used in the pipeline after processing. Type should be `VAE`.
        - `clip`: Outputs the CLIP model used in the pipeline after processing. Type should be `CLIP`.
        - `seed`: Outputs the seed used for random number generation in the pipeline after processing. Type should be `INT`.
        - `width`: unknown Type should be `INT`.
        - `height`: unknown Type should be `INT`.
        - `pos_string`: Outputs the final positive conditioning string after processing. Type should be `STRING`.
        - `neg_string`: Outputs the final negative conditioning string after processing. Type should be `STRING`.
