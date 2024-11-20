---
tags:
- Loader
- Model
- ModelIO
- ModelLoader
---

# EasyLoader (Zero123)
## Documentation
- Class name: `easy zero123Loader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The easy zero123Loader node is designed to simplify the process of loading and configuring 3D models for use within the ComfyUI framework. It abstracts away the complexities involved in setting up 3D model parameters, making it easier for users to integrate 3D content into their applications.
## Input types
### Required
- **`ckpt_name`**
    - Identifies the checkpoint name for the model to be loaded, serving as a key identifier for selecting the specific 3D model configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - Specifies the name of the VAE model to be used in conjunction with the 3D model, facilitating the generation or manipulation of 3D content.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`init_image`**
    - Provides an initial image to guide the generation or manipulation process, acting as a starting point for the 3D model's output.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Defines the width of the latent space to be used for generating or manipulating 3D content, impacting the resolution and detail of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Specifies the height of the latent space, influencing the vertical resolution and detail of the 3D model's output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Determines the number of instances to be processed in a single batch, affecting the efficiency and speed of the loading operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`elevation`**
    - Sets the elevation angle for the 3D model's viewpoint, altering the vertical perspective of the rendered content.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`azimuth`**
    - Defines the azimuth angle, adjusting the horizontal orientation and perspective of the 3D model's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The pipeline object configured with the specified 3D model and settings, ready for use in generating or manipulating 3D content.
    - Python dtype: `Pipeline`
- **`model`**
    - Comfy dtype: `MODEL`
    - The loaded 3D model object, prepared for integration and use within the ComfyUI framework.
    - Python dtype: `Model`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model associated with the 3D model, enabling advanced content generation and manipulation capabilities.
    - Python dtype: `VAE`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class zero123Loader:

    @classmethod
    def INPUT_TYPES(cls):
        def get_file_list(filenames):
            return [file for file in filenames if file != "put_models_here.txt" and "zero123" in file.lower()]

        return {"required": {
            "ckpt_name": (get_file_list(folder_paths.get_filename_list("checkpoints")),),
            "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

            "init_image": ("IMAGE",),
            "empty_latent_width": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
            "empty_latent_height": ("INT", {"default": 256, "min": 16, "max": MAX_RESOLUTION, "step": 8}),

            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),

            "elevation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
            "azimuth": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0}),
        },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def adv_pipeloader(self, ckpt_name, vae_name, init_image, empty_latent_width, empty_latent_height, batch_size, elevation, azimuth, prompt=None, my_unique_id=None):
        model: ModelPatcher | None = None
        vae: VAE | None = None
        clip: CLIP | None = None
        clip_vision = None

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        model, clip, vae, clip_vision = easyCache.load_checkpoint(ckpt_name, "Default", True)

        output = clip_vision.encode_image(init_image)
        pooled = output.image_embeds.unsqueeze(0)
        pixels = comfy.utils.common_upscale(init_image.movedim(-1, 1), empty_latent_width, empty_latent_height, "bilinear", "center").movedim(1, -1)
        encode_pixels = pixels[:, :, :, :3]
        t = vae.encode(encode_pixels)
        cam_embeds = camera_embeddings(elevation, azimuth)
        cond = torch.cat([pooled, cam_embeds.repeat((pooled.shape[0], 1, 1))], dim=-1)

        positive = [[cond, {"concat_latent_image": t}]]
        negative = [[torch.zeros_like(pooled), {"concat_latent_image": torch.zeros_like(t)}]]
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8])
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
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": 0,
                                    }
                }

        return (pipe, model, vae)

```
