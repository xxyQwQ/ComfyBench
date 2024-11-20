---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (SVD)
## Documentation
- Class name: `easy svdLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The `easy svdLoader` node is designed to facilitate the loading of SVD (Singular Value Decomposition) models into the system. It streamlines the process of identifying and retrieving SVD model files from a specified directory, ensuring that only relevant files are selected for further processing. This node plays a crucial role in simplifying the integration and utilization of SVD models within the broader computational framework, making it easier for users to leverage SVD-based functionalities.
## Input types
### Required
- **`ckpt_name`**
    - The `ckpt_name` parameter specifies the checkpoint name for the SVD model to be loaded. It is crucial for identifying the specific model file to be used in the loading process, thereby enabling the precise application of SVD models within the system.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - The `vae_name` parameter identifies the VAE model to be used alongside the SVD model, facilitating a combined application of these models for enhanced processing capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name`**
    - The `clip_name` parameter specifies the CLIP model to be used in conjunction with the SVD model, enabling advanced feature extraction and analysis.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`init_image`**
    - The `init_image` parameter is used to provide an initial image for processing, serving as a starting point for model applications.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`resolution`**
    - The `resolution` parameter specifies the desired resolution for the output, impacting the quality and size of the generated content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Specifies the width of the empty latent space to be used, affecting the dimensions of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Specifies the height of the empty latent space, influencing the vertical dimension of the generated content.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`video_frames`**
    - Determines the number of frames for video generation, directly affecting the video's length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`motion_bucket_id`**
    - Identifies the motion bucket to be used, influencing the motion characteristics of the generated video.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fps`**
    - Sets the frames per second for the video, impacting the playback speed and smoothness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`augmentation_level`**
    - Controls the level of augmentation applied, affecting the variety and intensity of visual modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`optional_positive`**
    - Optional positive conditioning text to guide the generation towards specific attributes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`optional_negative`**
    - Optional negative conditioning text to steer the generation away from certain attributes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The `pipe` output represents the pipeline configuration resulting from the loaded models, integrating SVD, VAE, and possibly other models.
    - Python dtype: `object`
- **`model`**
    - Comfy dtype: `MODEL`
    - The `model` output refers to the specific SVD model that has been loaded and is ready for use within the pipeline.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - The `vae` output indicates the VAE model that has been loaded and integrated into the system for combined use with the SVD model.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class svdLoader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "svd" in file.lower()]

        return {"required": {
                "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_name": (["None"] + folder_paths.get_filename_list("clip"),),

                "init_image": ("IMAGE",),
                "resolution": (resolution_strings, {"default": "1024 x 576"}),
                "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

                "video_frames": ("INT", {"default": 14, "min": 1, "max": 4096}),
                "motion_bucket_id": ("INT", {"default": 127, "min": 1, "max": 1023}),
                "fps": ("INT", {"default": 6, "min": 1, "max": 1024}),
                "augmentation_level": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01})
            },
            "optional": {
                "optional_positive": ("STRING", {"default": "", "multiline": True}),
                "optional_negative": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_name, init_image, resolution, empty_latent_width, empty_latent_height, video_frames, motion_bucket_id, fps, augmentation_level, optional_positive=None, optional_negative=None, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

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

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        if augmentation_level > 0:
            encode_pixels += torch.randn_like(pixels) * augmentation_level
        t = vae.encode(encode_pixels)
        positive = [[pooled,
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled),
                     {"motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level,
                      "concat_latent_image": torch.zeros_like(t)}]]
        if optional_positive is not None and optional_positive != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when positive is not empty")
            clip = easyCache.load_clip(clip_name)
            if has_chinese(optional_positive):
                optional_positive = zh_to_en([optional_positive])[0]
            positive_embeddings_final, = CLIPTextEncode().encode(clip, optional_positive)
            positive, = ConditioningConcat().concat(positive, positive_embeddings_final)
        if optional_negative is not None and optional_negative != '':
            if clip_name == 'None':
                raise Exception("You need choose a open_clip model when negative is not empty")
            if has_chinese(optional_negative):
                optional_positive = zh_to_en([optional_negative])[0]
            negative_embeddings_final, = CLIPTextEncode().encode(clip, optional_negative)
            negative, = ConditioningConcat().concat(negative, negative_embeddings_final)

        latent = torch.zeros([video_frames, 4, empty_latent_height // 8, empty_latent_width // 8])
        samples = {"samples": latent}

        image = easySampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive,
                "negative": negative,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": 0,

                "loader_settings": {"ckpt_name": ckpt_name,
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
