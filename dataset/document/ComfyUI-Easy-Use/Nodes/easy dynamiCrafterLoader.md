---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (DynamiCrafter)
## Documentation
- Class name: `easy dynamiCrafterLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

This node is designed to load and initialize the DynamiCrafter model, a specialized model for crafting dynamic content. It encapsulates the complexities of loading model configurations, initializing the model with the appropriate device settings, and preparing it for inference tasks.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the DynamiCrafter model to be loaded. This is crucial for identifying the correct model configuration and ensuring the model is loaded with the appropriate parameters.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Determines the number of clipping operations to skip during the model's inference process. This parameter can affect the speed and quality of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`init_image`**
    - The initial image to be used by the model, serving as a starting point for content generation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`resolution`**
    - Defines the resolution for the output content, impacting the detail and quality of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Tuple[int, int]`
- **`empty_latent_width`**
    - Specifies the width of the empty latent space to be used by the model, affecting the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Specifies the height of the empty latent space to be used by the model, affecting the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive`**
    - A string of positive prompts or keywords that guide the content generation towards desired attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative`**
    - A string of negative prompts or keywords that guide the content generation away from certain attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`use_interpolate`**
    - Indicates whether interpolation should be used in the content generation process, affecting the smoothness and continuity of dynamic content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`fps`**
    - Specifies the frames per second for the generated content, impacting the smoothness and playback speed of dynamic videos.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`frames`**
    - Determines the total number of frames to be generated, defining the length of the dynamic content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale_latents`**
    - Indicates whether the latent space should be scaled, affecting the variation and diversity of the generated content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`optional_vae`**
    - An optional parameter to specify a VAE model to be used in conjunction with the DynamiCrafter model for enhanced content generation.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Provides the main pipeline object for content generation.
    - Python dtype: `Pipeline`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the initialized DynamiCrafter model ready for generating dynamic content.
    - Python dtype: `Model`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE model associated with the DynamiCrafter, used for variational autoencoder tasks.
    - Python dtype: `VAE`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class dynamiCrafterLoader(DynamiCrafter):

    def __init__(self):
        super().__init__()

    @classmethod
    def INPUT_TYPES(cls):

        return {"required": {
                "model_name": (list(DYNAMICRAFTER_MODELS.keys()),),
                "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                "init_image": ("IMAGE",),
                "resolution": (resolution_strings, {"default": "512 x 512"}),
                "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": True}),

                "use_interpolate": ("BOOLEAN", {"default": False}),
                "fps": ("INT", {"default": 15, "min": 1, "max": 30, "step": 1},),
                "frames": ("INT", {"default": 16}),
                "scale_latents": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "optional_vae": ("VAE",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def get_clip_file(self, node_name):
        clip_list = folder_paths.get_filename_list("clip")
        pattern = 'sd2-1-open-clip|model\.(safetensors|bin)$'
        clip_files = [e for e in clip_list if re.search(pattern, e, re.IGNORECASE)]

        clip_name = clip_files[0] if len(clip_files)>0 else None
        clip_file = folder_paths.get_full_path("clip", clip_name) if clip_name else None
        if clip_name is not None:
            log_node_info(node_name, f"Using {clip_name}")

        return clip_file, clip_name

    def get_clipvision_file(self, node_name):
        clipvision_list = folder_paths.get_filename_list("clip_vision")
        pattern = '(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model|open_clip_pytorch_model\.(bin|safetensors))'
        clipvision_files = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

        clipvision_name = clipvision_files[0] if len(clipvision_files)>0 else None
        clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_name) if clipvision_name else None
        if clipvision_name is not None:
            log_node_info(node_name, f"Using {clipvision_name}")

        return clipvision_file, clipvision_name

    def get_vae_file(self, node_name):
        vae_list = folder_paths.get_filename_list("vae")
        pattern = 'vae-ft-mse-840000-ema-pruned\.(pt|bin|safetensors)$'
        vae_files = [e for e in vae_list if re.search(pattern, e, re.IGNORECASE)]

        vae_name = vae_files[0] if len(vae_files)>0 else None
        vae_file = folder_paths.get_full_path("vae", vae_name) if vae_name else None
        if vae_name is not None:
            log_node_info(node_name, f"Using {vae_name}")

        return vae_file, vae_name

    def adv_pipeloader(self, model_name, clip_skip, init_image, resolution, empty_latent_width, empty_latent_height, positive, negative, use_interpolate, fps, frames, scale_latents, optional_vae=None, prompt=None, my_unique_id=None):
        positive_embeddings_final, negative_embeddings_final = None, None
        # resolution
        if resolution != "自定义 x 自定义":
            try:
                width, height = map(int, resolution.split(' x '))
                empty_latent_width = width
                empty_latent_height = height
            except ValueError:
                raise ValueError("Invalid base_resolution format.")

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        models_0 = list(DYNAMICRAFTER_MODELS.keys())[0]

        if optional_vae:
            vae = optional_vae
            vae_name = None
        else:
            vae_file, vae_name = self.get_vae_file("easy dynamiCrafterLoader")
            if vae_file is None:
                vae_name = "vae-ft-mse-840000-ema-pruned.safetensors"
                get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['vae_url'], os.path.join(folder_paths.models_dir, "vae"),
                                   vae_name)
            vae = easyCache.load_vae(vae_name)

        clip_file, clip_name = self.get_clip_file("easy dynamiCrafterLoader")
        if clip_file is None:
            clip_name = 'sd2-1-open-clip.safetensors'
            get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['clip_url'], os.path.join(folder_paths.models_dir, "clip"),
                           clip_name)

        clip = easyCache.load_clip(clip_name)
        # load clip vision
        clip_vision_file, clip_vision_name = self.get_clipvision_file("easy dynamiCrafterLoader")
        if clip_vision_file is None:
            clip_vision_name = 'CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors'
            clip_vision_file = get_local_filepath(DYNAMICRAFTER_MODELS[models_0]['clip_vision_url'], os.path.join(folder_paths.models_dir, "clip_vision"),
                                   clip_vision_name)
        clip_vision = load_clip_vision(clip_vision_file)
        # load unet model
        model_path = get_local_filepath(DYNAMICRAFTER_MODELS[model_name]['model_url'], DYNAMICRAFTER_DIR)
        model_patcher, image_proj_model = self.load_dynamicrafter(model_path)

        # rescale cfg

        # apply
        model, empty_latent, image_latent = self.process_image_conditioning(model_patcher, clip_vision, vae, image_proj_model, init_image, use_interpolate, fps, frames, scale_latents)

        clipped = clip.clone()
        if clip_skip != 0:
            clipped.clip_layer(clip_skip)

        if positive is not None and positive != '':
            if has_chinese(positive):
                positive = zh_to_en([positive])[0]
            positive_embeddings_final, = CLIPTextEncode().encode(clipped, positive)
        if negative is not None and negative != '':
            if has_chinese(negative):
                negative = zh_to_en([negative])[0]
            negative_embeddings_final, = CLIPTextEncode().encode(clipped, negative)

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "vae": vae,
                "clip": clip,
                "clip_vision": clip_vision,

                "samples": empty_latent,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": model_name,
                                    "vae_name": vae_name,

                                    "positive": positive,
                                    "negative": negative,
                                    "resolution": resolution,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": 1,
                                    "seed": 0,
                                     }
                }

        return (pipe, model, vae)

```
