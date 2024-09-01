- `ttN pipeLoader_v2`: This node is designed to load and initialize pipelines for various tasks within the tinyterraNodes framework, facilitating the setup and configuration of data processing and model interaction pipelines.
    - Parameters:
        - `ckpt_name`: Specifies the checkpoint name for model initialization, allowing for the selection of different model states. Type should be `COMBO[STRING]`.
        - `config_name`: Determines the configuration name, enabling the selection from various predefined settings. Type should be `COMBO[STRING]`.
        - `vae_name`: Specifies the VAE model name, allowing for the selection of different VAE models for processing. Type should be `COMBO[STRING]`.
        - `clip_skip`: An integer value to skip certain CLIP model layers, optimizing performance based on specific requirements. Type should be `INT`.
        - `loras`: A string specifying LoRA parameters, enabling dynamic adjustment of LoRA layers for model refinement. Type should be `STRING`.
        - `positive`: unknown Type should be `STRING`.
        - `positive_token_normalization`: unknown Type should be `COMBO[STRING]`.
        - `positive_weight_interpretation`: unknown Type should be `COMBO[STRING]`.
        - `negative`: unknown Type should be `STRING`.
        - `negative_token_normalization`: unknown Type should be `COMBO[STRING]`.
        - `negative_weight_interpretation`: unknown Type should be `COMBO[STRING]`.
        - `empty_latent_aspect`: Defines the aspect ratio for empty latent space initialization, setting the groundwork for image generation. Type should be `COMBO[STRING]`.
        - `empty_latent_width`: Specifies the width for empty latent space, determining the initial dimensions for image generation. Type should be `INT`.
        - `empty_latent_height`: Specifies the height for empty latent space, determining the initial dimensions for image generation. Type should be `INT`.
        - `seed`: An integer seed for random number generation, ensuring reproducibility across pipeline executions. Type should be `INT`.
        - `prepend_positive`: Optional text to prepend to positive conditioning, enriching the context for desired attributes. Type should be `STRING`.
        - `prepend_negative`: Optional text to prepend to negative conditioning, refining the context for undesired attributes. Type should be `STRING`.
    - Inputs:
        - `model_override`: Allows for the override of the default model, enabling the use of alternative models within the pipeline. Type should be `MODEL`.
        - `clip_override`: Allows for the override of the default CLIP model, integrating alternative visual understanding capabilities. Type should be `CLIP`.
        - `optional_lora_stack`: Optional parameter to specify a stack of LoRA adjustments, enhancing model performance with custom configurations. Type should be `LORA_STACK`.
        - `optional_controlnet_stack`: Optional parameter to specify a stack of ControlNet adjustments, enabling fine-tuned control over model behavior. Type should be `CONTROLNET_STACK`.
    - Outputs:
        - `pipe`: The updated pipeline configuration after processing inputs. Type should be `PIPE_LINE`.
        - `model`: unknown Type should be `MODEL`.
        - `positive`: The processed positive conditioning, ready for use in the pipeline. Type should be `CONDITIONING`.
        - `negative`: The processed negative conditioning, ready for use in the pipeline. Type should be `CONDITIONING`.
        - `latent`: unknown Type should be `LATENT`.
        - `vae`: unknown Type should be `VAE`.
        - `clip`: The CLIP model parameter, integrated based on the optional input. Type should be `CLIP`.
        - `seed`: unknown Type should be `INT`.
        - `width`: unknown Type should be `INT`.
        - `height`: unknown Type should be `INT`.
        - `pos_string`: unknown Type should be `STRING`.
        - `neg_string`: unknown Type should be `STRING`.