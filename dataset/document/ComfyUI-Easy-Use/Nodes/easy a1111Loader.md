---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (A1111)
## Documentation
- Class name: `easy a1111Loader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The easy a1111Loader node is designed to simplify the process of loading and configuring models for use in AI-driven applications. It abstracts the complexities involved in setting up models, including checkpoint loading, LoRA (Low-Rank Adaptation) adjustments, and optional control net stacking, making it easier for users to get started with their AI projects.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the model to be loaded. It is crucial for identifying which pre-trained model to use, affecting the node's execution by determining the initial state of the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - Defines the name of the VAE (Variational Autoencoder) to be used alongside the model. This parameter influences the quality and characteristics of the generated content by specifying the underlying generative model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Specifies the number of CLIP model layers to skip during processing, allowing for customization of the model's focus and performance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`lora_name`**
    - Indicates the name of the LoRA to apply to the model, allowing for fine-tuning and adaptation of the model's weights without extensive retraining. This parameter is key for customizing the model's behavior for specific tasks.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_model_strength`**
    - Determines the strength of the LoRA model adaptation, directly impacting the model's performance and output by adjusting the intensity of the applied modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_clip_strength`**
    - Sets the strength of the LoRA clip adaptation, affecting the final output by modifying how strongly the clip adaptation influences the model's behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resolution`**
    - Specifies the resolution for the output, directly affecting the visual quality and detail of the generated images or content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Defines the width of the empty latent space, influencing the dimensionality and potential complexity of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Sets the height of the empty latent space, impacting the size and detail level of the generated outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive`**
    - A string input for positive prompts, guiding the model to emphasize or generate content that aligns with the specified attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative`**
    - A string input for negative prompts, instructing the model to de-emphasize or avoid content related to the specified attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`batch_size`**
    - Determines the number of instances to process in a single batch, affecting the efficiency and speed of model execution.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_lora_stack`**
    - Allows for the specification of an optional stack of LoRA adjustments, enabling more complex and layered model customizations.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `List[Dict[str, Any]]`
- **`optional_controlnet_stack`**
    - Enables the inclusion of an optional control net stack for further model customization and fine-tuning, offering advanced control over the model's behavior.
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `List[Dict[str, Any]]`
- **`a1111_prompt_style`**
    - A boolean flag to toggle the A1111 prompt style, affecting how prompts are interpreted and processed by the model.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the configured pipeline, encapsulating the entire setup process for immediate use.
    - Python dtype: `Any`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the loaded and configured model, ready for use in generating or processing content.
    - Python dtype: `Any`
- **`vae`**
    - Comfy dtype: `VAE`
    - Provides the loaded VAE model, essential for generating or manipulating content in the latent space.
    - Python dtype: `Any`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class a1111Loader(fullLoader):
    @classmethod
    def INPUT_TYPES(cls):
        a1111_prompt_style_default = False
        checkpoints = folder_paths.get_filename_list("checkpoints")
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {
            "required": {
                "ckpt_name": (checkpoints,),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                "lora_name": (loras,),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default":"", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default":"", "placeholder": "Negative", "multiline": True}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                "a1111_prompt_style": ("BOOLEAN", {"default": a1111_prompt_style_default}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "a1111loader"
    CATEGORY = "EasyUse/Loaders"

    def a1111loader(self, ckpt_name, vae_name, clip_skip,
                       lora_name, lora_model_strength, lora_clip_strength,
                       resolution, empty_latent_width, empty_latent_height,
                       positive, negative, batch_size, optional_lora_stack=None, optional_controlnet_stack=None, a1111_prompt_style=False, prompt=None,
                       my_unique_id=None):

        return super().adv_pipeloader(ckpt_name, 'Default', vae_name, clip_skip,
             lora_name, lora_model_strength, lora_clip_strength,
             resolution, empty_latent_width, empty_latent_height,
             positive, 'mean', 'A1111',
             negative,'mean','A1111',
             batch_size, None, None,  None, optional_lora_stack=optional_lora_stack, optional_controlnet_stack=optional_controlnet_stack,a1111_prompt_style=a1111_prompt_style, prompt=prompt,
             my_unique_id=my_unique_id
        )

```
