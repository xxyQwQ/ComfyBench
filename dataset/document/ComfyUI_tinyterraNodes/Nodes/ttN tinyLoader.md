---
tags:
- Face
---

# tinyLoader
## Documentation
- Class name: `ttN tinyLoader`
- Category: `🌏 tinyterra/base`
- Output node: `False`

The ttN tinyLoader node is designed for loading and initializing tiny models within the ComfyUI framework, facilitating the integration and management of lightweight models for efficient processing and analysis.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for loading the model, crucial for initializing the model with pre-trained weights.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_name`**
    - Determines the configuration name for the model, essential for setting up model parameters and behaviors.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sampling`**
    - Defines the sampling strategy to be used by the model, affecting the generation or processing outcomes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`zsnr`**
    - Specifies the zero-shot noise ratio, influencing the model's inference behavior.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `float`
- **`cfg_rescale_mult`**
    - Determines the CFG rescale multiplier, adjusting the model's configuration for specific tasks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`vae_name`**
    - Specifies the VAE name for loading, essential for initializing the model with a specific variational autoencoder.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Defines the number of layers to skip in the CLIP model, affecting the model's vision processing capabilities.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_aspect`**
    - Specifies the aspect ratio for generating an empty latent space, crucial for initializing the model's latent representation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Determines the width of the empty latent space, essential for setting up the model's latent representation size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Specifies the height of the empty latent space, crucial for defining the model's latent representation dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_size`**
    - Defines the batch size for processing, affecting the model's performance and resource utilization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified or initialized model ready for further processing or analysis within the ComfyUI framework.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The generated or modified latent representation, ready for further processing or analysis.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - The loaded or initialized VAE model, essential for variational autoencoder tasks.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The loaded or initialized CLIP model, crucial for vision and language processing tasks.
    - Python dtype: `str`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the generated or processed output, reflecting the dimensions of the model's output space.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the generated or processed output, indicating the dimensions of the model's output space.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_tinyLoader:
    version = '1.1.0'
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["width x height [custom]",
                        "512 x 512 [S] 1:1",
                        "768 x 768 [S] 1:1",
                        "910 x 910 [S] 1:1",

                        "512 x 682 [P] 3:4",
                        "512 x 768 [P] 2:3",
                        "512 x 910 [P] 9:16",

                        "682 x 512 [L] 4:3",
                        "768 x 512 [L] 3:2",
                        "910 x 512 [L] 16:9",
                        
                        "1024 x 1024 [S] 1:1",                        
                        "512 x 1024 [P] 1:2",
                        "1024 x 512 [L] 2:1",

                        "640 x 1536 [P] 9:21",
                        "704 x 1472 [P] 9:19",
                        "768 x 1344 [P] 9:16",
                        "768 x 1216 [P] 5:8",
                        "832 x 1216 [P] 2:3",
                        "896 x 1152 [P] 3:4",

                        "1536 x 640 [L] 21:9",
                        "1472 x 704 [L] 19:9",
                        "1344 x 768 [L] 16:9",
                        "1216 x 768 [L] 8:5",
                        "1216 x 832 [L] 3:2",
                        "1152 x 896 [L] 4:3",
                        ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "sampling": (["Default", "eps", "v_prediction", "lcm", "x0"], {"default": "Default"}),
                        "zsnr": ("BOOLEAN", {"default": False}),
                        "cfg_rescale_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),
                        
                        "empty_latent_aspect": (aspect_ratios, {"default":"512 x 512 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_tinyLoader.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("MODEL", "LATENT", "VAE", "CLIP", "INT", "INT",)
    RETURN_NAMES = ("model", "latent", "vae", "clip", "width", "height",)

    FUNCTION = "miniloader"
    CATEGORY = "🌏 tinyterra/base"

    def miniloader(self, ckpt_name, config_name, sampling, zsnr, cfg_rescale_mult, vae_name, clip_skip,
                       empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size,
                       prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height)
        samples = {"samples": latent}

        model, clip, vae = loader.load_checkpoint(ckpt_name, config_name, clip_skip)
        if vae_name != "Baked VAE":
            vae = loader.load_vae(vae_name)

        if sampling != "Default":
            MSD = comfy_extras.nodes_model_advanced.ModelSamplingDiscrete()
            model = MSD.patch(model, sampling, zsnr)[0]

        if cfg_rescale_mult > 0:
            CFGR = comfy_extras.nodes_model_advanced.RescaleCFG()
            model = CFGR.patch(model, cfg_rescale_mult)[0]

        return (model, samples, vae, clip, empty_latent_width, empty_latent_height)

```
