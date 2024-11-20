---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (Comfy)
## Documentation
- Class name: `easy comfyLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The `easy comfyLoader` node is designed to simplify the process of loading and configuring models for image generation tasks. It abstracts away the complexities involved in setting up models like Stable Diffusion, allowing users to easily specify model checkpoints, VAEs, and other parameters through a user-friendly interface. This node aims to make advanced image generation accessible to a wider audience by providing an easy-to-use loading mechanism.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the model to be loaded. This parameter is crucial for determining which pre-trained model will be used for image generation, directly affecting the output quality and style.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`vae_name`**
    - Defines the VAE (Variational Autoencoder) to be used in conjunction with the specified model checkpoint. The choice of VAE can influence the characteristics of the generated images, such as their clarity and style.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Determines the number of frames to skip when processing video input with CLIP, allowing for customization of the temporal sampling rate.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`lora_name`**
    - Specifies the LoRA model to be applied for enhancing the generation process, offering a way to adjust the model's behavior and output through learned residual adapters.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_model_strength`**
    - Defines the strength of the LoRA model adjustments, allowing for fine-tuning of the model's influence on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_clip_strength`**
    - Specifies the strength of the CLIP adjustments when using a LoRA model, enabling precise control over the influence of CLIP-guided enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resolution`**
    - Sets the resolution for the generated images, directly impacting the visual quality and detail.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Specifies the width of the empty latent space to be used for image generation, affecting the aspect ratio and size of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Defines the height of the empty latent space, influencing the aspect ratio and dimensions of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive`**
    - A positive text prompt that guides the image generation towards desired themes or concepts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative`**
    - A negative text prompt used to steer the image generation away from certain themes or concepts.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`batch_size`**
    - Determines the number of images to be generated in a single batch, affecting the efficiency and speed of the generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_lora_stack`**
    - Optionally specifies a stack of LoRA models to be applied, offering enhanced customization for the generation process.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `str`
- **`optional_controlnet_stack`**
    - Optionally specifies a stack of ControlNet models for advanced control over the generation process.
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs a configured pipeline object ready for image generation, encapsulating the loaded models and settings.
    - Python dtype: `Dict[str, Any]`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the main model object loaded and configured by the node, ready for use in generation tasks.
    - Python dtype: `ModelPatcher | None`
- **`vae`**
    - Comfy dtype: `VAE`
    - Provides the loaded VAE model object, which is essential for processing and generating the final image outputs.
    - Python dtype: `VAE | None`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class comfyLoader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {"optional_lora_stack": ("LORA_STACK",), "optional_controlnet_stack": ("CONTROL_NET_STACK",),},
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "comfyloader"
    CATEGORY = "EasyUse/Loaders"

    def comfyloader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, prompt=None,
                      my_unique_id=None):
        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'none', 'comfy',
             negative, 'none', 'comfy',
             batch_size, None, None, None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack, a1111_prompt_style=False, prompt=prompt,
             my_unique_id=my_unique_id
         )

```
