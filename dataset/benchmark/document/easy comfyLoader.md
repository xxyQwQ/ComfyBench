- `easy comfyLoader`: The `easy comfyLoader` node is designed to simplify the process of loading and configuring models for image generation tasks. It abstracts away the complexities involved in setting up models like Stable Diffusion, allowing users to easily specify model checkpoints, VAEs, and other parameters through a user-friendly interface. This node aims to make advanced image generation accessible to a wider audience by providing an easy-to-use loading mechanism.
    - Inputs:
        - `ckpt_name` (Required): Specifies the checkpoint name for the model to be loaded. This parameter is crucial for determining which pre-trained model will be used for image generation, directly affecting the output quality and style. Type should be `COMBO[STRING]`.
        - `vae_name` (Required): Defines the VAE (Variational Autoencoder) to be used in conjunction with the specified model checkpoint. The choice of VAE can influence the characteristics of the generated images, such as their clarity and style. Type should be `COMBO[STRING]`.
        - `clip_skip` (Required): Determines the number of frames to skip when processing video input with CLIP, allowing for customization of the temporal sampling rate. Type should be `INT`.
        - `lora_name` (Required): Specifies the LoRA model to be applied for enhancing the generation process, offering a way to adjust the model's behavior and output through learned residual adapters. Type should be `COMBO[STRING]`.
        - `lora_model_strength` (Required): Defines the strength of the LoRA model adjustments, allowing for fine-tuning of the model's influence on the generation process. Type should be `FLOAT`.
        - `lora_clip_strength` (Required): Specifies the strength of the CLIP adjustments when using a LoRA model, enabling precise control over the influence of CLIP-guided enhancements. Type should be `FLOAT`.
        - `resolution` (Required): Sets the resolution for the generated images, directly impacting the visual quality and detail. Type should be `COMBO[STRING]`.
        - `empty_latent_width` (Required): Specifies the width of the empty latent space to be used for image generation, affecting the aspect ratio and size of the output. Type should be `INT`.
        - `empty_latent_height` (Required): Defines the height of the empty latent space, influencing the aspect ratio and dimensions of the generated images. Type should be `INT`.
        - `positive` (Required): A positive text prompt that guides the image generation towards desired themes or concepts. Type should be `STRING`.
        - `negative` (Required): A negative text prompt used to steer the image generation away from certain themes or concepts. Type should be `STRING`.
        - `batch_size` (Required): Determines the number of images to be generated in a single batch, affecting the efficiency and speed of the generation process. Type should be `INT`.
        - `optional_lora_stack` (Optional): Optionally specifies a stack of LoRA models to be applied, offering enhanced customization for the generation process. Type should be `LORA_STACK`.
        - `optional_controlnet_stack` (Optional): Optionally specifies a stack of ControlNet models for advanced control over the generation process. Type should be `CONTROL_NET_STACK`.
    - Outputs:
        - `pipe`: Outputs a configured pipeline object ready for image generation, encapsulating the loaded models and settings. Type should be `PIPE_LINE`.
        - `model`: Returns the main model object loaded and configured by the node, ready for use in generation tasks. Type should be `MODEL`.
        - `vae`: Provides the loaded VAE model object, which is essential for processing and generating the final image outputs. Type should be `VAE`.