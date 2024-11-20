---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (Full)
## Documentation
- Class name: `easy fullLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The `easy fullLoader` node is designed to streamline the process of loading and initializing various models and components required for image generation tasks. It encapsulates the complexity of handling different model types, such as Stable Diffusion, control nets, and DIT loaders, providing a unified interface for easy integration and use within the ComfyUI framework. This node significantly simplifies the setup and configuration phase, enabling users to focus on the creative aspects of image generation.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the model to be loaded. This parameter is crucial for identifying and retrieving the correct model version, affecting the quality and characteristics of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_name`**
    - Specifies the configuration name for further customization of the loading process, allowing for more tailored model initialization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - Determines the VAE model to be used in conjunction with the primary model. The choice of VAE can influence the style and details of the generated images, making it an important aspect of the configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Determines the number of layers to skip in the CLIP model, affecting the depth of text encoding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`lora_name`**
    - Identifies the LoRA model to be applied, if any. LoRA (Low-Rank Adaptation) allows for fine-tuning and adjusting model behaviors without extensive retraining, offering a way to customize the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_model_strength`**
    - Controls the intensity of the LoRA model's influence on the generation process. This parameter allows users to balance between the original model's output and the modifications introduced by LoRA.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_clip_strength`**
    - Adjusts the strength of LoRA's effect specifically on the CLIP model's outputs. This fine-tuning capability enables more precise control over the semantic alignment between text prompts and generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resolution`**
    - Sets the resolution for the generated images. Higher resolutions produce more detailed images but require more computational resources.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `int`
- **`empty_latent_width`**
    - Specifies the width of the empty latent space to be used for image generation. This parameter is part of the setup for generating images from scratch or modifying existing ones.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Defines the height of the empty latent space, complementing the width setting to determine the overall dimensions of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive`**
    - A text prompt that guides the model towards generating images with desired attributes. Positive prompts encourage the inclusion of specific elements or themes in the output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`positive_token_normalization`**
    - Defines the method for normalizing positive prompt tokens, influencing the weighting of text prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive_weight_interpretation`**
    - Specifies how the weights of positive prompts are interpreted, affecting the emphasis on certain aspects of the image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative`**
    - A text prompt used to steer the model away from certain attributes or themes. Negative prompts help in excluding undesired elements from the generated images.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_token_normalization`**
    - Defines the method for normalizing negative prompt tokens, influencing the weighting of text prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative_weight_interpretation`**
    - Specifies how the weights of negative prompts are interpreted, affecting the emphasis on certain aspects of the image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`batch_size`**
    - Determines the number of images to be generated in a single batch. Adjusting the batch size can impact performance and resource utilization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`model_override`**
    - Allows for the override of the default model with a specified one, offering flexibility in model selection.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`clip_override`**
    - Enables the override of the default CLIP model with a specified one, allowing for customization of text encoding.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`vae_override`**
    - Permits the override of the default VAE model with a specified one, providing options for style and detail customization.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`optional_lora_stack`**
    - Optional parameter to specify a stack of LoRA models for enhanced customization.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `str`
- **`optional_controlnet_stack`**
    - Optional parameter to specify a stack of control net models for additional adjustments to the generation process.
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `str`
- **`a1111_prompt_style`**
    - Enables the use of a specific prompt style, offering further customization of the text-to-image translation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the configured pipeline necessary for generating images based on the provided settings.
    - Python dtype: `Dict`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the loaded model after configuration, ready for image generation.
    - Python dtype: `Any`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the loaded VAE model, which influences the style and details of the generated images.
    - Python dtype: `Any`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the loaded CLIP model, used for encoding text prompts into visual concepts.
    - Python dtype: `Any`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive embeddings generated from the text prompts, guiding the image generation towards desired attributes.
    - Python dtype: `torch.Tensor`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative embeddings generated from the text prompts, steering the image generation away from certain attributes.
    - Python dtype: `torch.Tensor`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Provides the latent space representation used for image generation, offering a foundation for creativity.
    - Python dtype: `Any`
- **`ui`**
    - Provides a user interface component for dynamically adjusting positive and negative prompts, enhancing interactivity and user control over the image generation process.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class fullLoader:

    @classmethod
    def INPUT_TYPES(cls):
        a1111_prompt_style_default = False

        return {"required": {
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
            "config_name": (["Default", ] + folder_paths.get_filename_list("configs"), {"default": "Default"}),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
            "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

            "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
            "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

            "resolution": (resolution_strings,),
            "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

            "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
            "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
            "positive_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),
            "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
            "negative_weight_interpretation": (["comfy",  "A1111", "comfy++", "compel", "fixed attention"],),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
        },
            "optional": {"model_override": ("MODEL",), "clip_override": ("CLIP",), "vae_override": ("VAE",), "optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",), "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE", "CLIP", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("pipe", "model", "vae", "clip", "positive", "negative", "latent")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, positive_token_normalization, positive_weight_interpretation,
                       negative, negative_token_normalization, negative_weight_interpretation,
                       batch_size, model_override=None, clip_override=None, vae_override=None, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None
                       ):

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Load models
        log_node_warn("正在加载模型...")
        model, clip, vae, clip_vision, lora_stack = easyCache.load_main(ckpt_name, config_name, vae_name, lora_name, lora_model_strength, lora_clip_strength, optional_lora_stack, model_override, clip_override, vae_override, prompt)

        # Create Empty Latent
        sd3 = True if get_sd_version(model) == 'sd3' else False
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, sd3=sd3)

        # Prompt to Conditioning
        positive_embeddings_final, positive_wildcard_prompt, model, clip = prompt_to_cond('positive', model, clip, clip_skip, lora_stack, positive, positive_token_normalization, positive_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)
        negative_embeddings_final, negative_wildcard_prompt, model, clip = prompt_to_cond('negative', model, clip, clip_skip, lora_stack, negative, negative_token_normalization, negative_weight_interpretation, a1111_prompt_style, my_unique_id, prompt, easyCache)

        # Conditioning add controlnet
        if optional_controlnet_stack is not None and len(optional_controlnet_stack) > 0:
            for controlnet in optional_controlnet_stack:
                positive_embeddings_final, negative_embeddings_final = easyControlnet().apply(controlnet[0], controlnet[5], positive_embeddings_final, negative_embeddings_final, controlnet[1], start_percent=controlnet[2], end_percent=controlnet[3], control_net=None, scale_soft_weights=controlnet[4], mask=None, easyCache=easyCache, use_cache=True)

        log_node_warn("加载完毕...")
        pipe = {
            "model": model,
            "positive": positive_embeddings_final,
            "negative": negative_embeddings_final,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": None,

            "loader_settings": {
                "ckpt_name": ckpt_name,
                "vae_name": vae_name,
                "lora_name": lora_name,
                "lora_model_strength": lora_model_strength,
                "lora_clip_strength": lora_clip_strength,
                "lora_stack": lora_stack,

                "clip_skip": clip_skip,
                "a1111_prompt_style": a1111_prompt_style,
                "positive": positive,
                "positive_token_normalization": positive_token_normalization,
                "positive_weight_interpretation": positive_weight_interpretation,
                "negative": negative,
                "negative_token_normalization": negative_token_normalization,
                "negative_weight_interpretation": negative_weight_interpretation,
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {"positive": positive_wildcard_prompt, "negative": negative_wildcard_prompt}, "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}

```
