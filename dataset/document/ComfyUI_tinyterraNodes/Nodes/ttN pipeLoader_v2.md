---
tags:
- Image
- Pipeline
---

# pipeLoader
## Documentation
- Class name: `ttN pipeLoader_v2`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

The `ttN pipeLoader_v2` node is designed for advanced data loading and preprocessing within a pipeline. It focuses on efficiently handling and transforming data to prepare it for subsequent processing or analysis steps, leveraging enhanced capabilities over its predecessor for improved performance and flexibility.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for model loading, affecting the pipeline's configuration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_name`**
    - Specifies the configuration name for the pipeline, affecting how data is processed and managed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - Specifies the VAE model name for loading, affecting the pipeline's configuration and capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_skip`**
    - Determines the number of CLIP layers to skip, affecting the data preprocessing phase.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`loras`**
    - Specifies the LoRA models to be used, affecting the pipeline's performance and output.
    - Comfy dtype: `STRING`
    - Python dtype: `list`
- **`positive`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`positive_token_normalization`**
    - Indicates the method for normalizing positive tokens, affecting text processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive_weight_interpretation`**
    - Specifies how the weights for positive tokens are interpreted, affecting text processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`negative_token_normalization`**
    - Indicates the method for normalizing negative tokens, affecting text processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative_weight_interpretation`**
    - Specifies how the weights for negative tokens are interpreted, affecting text processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_aspect`**
    - Specifies the aspect ratio for empty latent images, affecting the pipeline's image processing capabilities.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`empty_latent_height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`batch_size`**
    - Specifies the batch size for processing, affecting the pipeline's efficiency and throughput.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - Specifies the seed for random number generation in the pipeline, ensuring reproducibility.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`model_override`**
    - Allows for overriding the default model, affecting the pipeline's flexibility and output.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`clip_override`**
    - Allows for overriding the default CLIP model, affecting the pipeline's flexibility and output.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`optional_lora_stack`**
    - Specifies an optional stack of LoRA models, enhancing the pipeline's adaptability and performance.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `list`
- **`optional_controlnet_stack`**
    - Specifies an optional stack of ControlNet models, enhancing the pipeline's adaptability and performance.
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `list`
- **`prepend_positive`**
    - Specifies text to prepend to positive conditioning, affecting text processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prepend_negative`**
    - Specifies text to prepend to negative conditioning, affecting text processing.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs a modified pipeline object, incorporating the loaded and preprocessed data ready for further stages.
    - Python dtype: `dict`
- **`model`**
    - Comfy dtype: `MODEL`
    - Outputs the model used in the pipeline after processing.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive conditioning used in the pipeline after processing.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative conditioning used in the pipeline after processing.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Outputs the latent variables used in the pipeline after processing.
    - Python dtype: `int`
- **`vae`**
    - Comfy dtype: `VAE`
    - Outputs the VAE model used in the pipeline after processing.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Outputs the CLIP model used in the pipeline after processing.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - Outputs the seed used for random number generation in the pipeline after processing.
    - Python dtype: `int`
- **`width`**
    - Comfy dtype: `INT`
    - unknown
    - Python dtype: `unknown`
- **`height`**
    - Comfy dtype: `INT`
    - unknown
    - Python dtype: `unknown`
- **`pos_string`**
    - Comfy dtype: `STRING`
    - Outputs the final positive conditioning string after processing.
    - Python dtype: `str`
- **`neg_string`**
    - Comfy dtype: `STRING`
    - Outputs the final negative conditioning string after processing.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeLoader_v2:
    version = '2.1.0'
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
                        
                        "512 x 1024 [P] 1:2",
                        "1024 x 512 [L] 2:1",
                        "1024 x 1024 [S] 1:1",
                        ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                        "loras": ("STRING", {"placeholder": "<lora:loraName:weight:optClipWeight>", "multiline": True}),

                        "positive": ("STRING", {"default": "Positive","multiline": True, "dynamicPrompts": True}),
                        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True, "dynamicPrompts": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "empty_latent_aspect": (aspect_ratios, {"default":"512 x 512 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },                
                "optional": {
                    "model_override": ("MODEL",), 
                    "clip_override": ("CLIP",), 
                    "optional_lora_stack": ("LORA_STACK",),
                    "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                    "prepend_positive": ("STRING", {"default": None, "forceInput": True}),
                    "prepend_negative": ("STRING", {"default": None, "forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoader_v2.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("PIPE_LINE" ,"MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("pipe","model", "positive", "negative", "latent", "vae", "clip", "seed", "width", "height", "pos_string", "neg_string")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "🌏 tinyterra/pipe"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       loras,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size, seed,
                       model_override=None, clip_override=None, optional_lora_stack=None, optional_controlnet_stack=None, prepend_positive=None, prepend_negative=None,
                       prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height)
        samples = {"samples":latent}

        loader.clear_cache(prompt)
        model, clip, vae = loader.load_main3(ckpt_name, config_name, vae_name, loras, clip_skip, model_override, clip_override, optional_lora_stack, my_unique_id)

        positive_embedding = loader.embedding_encode(positive, positive_token_normalization, positive_weight_interpretation, clip, seed=seed, title='pipeLoader Positive', my_unique_id=my_unique_id, prepend_text=prepend_positive)
        negative_embedding = loader.embedding_encode(negative, negative_token_normalization, negative_weight_interpretation, clip, seed=seed, title='pipeLoader Negative', my_unique_id=my_unique_id, prepend_text=prepend_negative)

        if optional_controlnet_stack is not None and len(optional_controlnet_stack) > 0:
            for cnt in optional_controlnet_stack:
                positive_embedding, negative_embedding = loader.load_controlNet(positive_embedding, negative_embedding, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4])

        image = None

        pipe = {"model": model,
                "positive": positive_embedding,
                "negative": negative_embedding,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": seed,

                "loader_settings": None,
        }

        final_positive = (prepend_positive + ' ' if prepend_positive else '') + (positive + ' ' if positive else '')
        final_negative = (prepend_negative + ' ' if prepend_negative else '') + (negative + ' ' if negative else '')

        return (pipe, model, positive_embedding, negative_embedding, samples, vae, clip, seed, empty_latent_width, empty_latent_height, final_positive, final_negative)

```
