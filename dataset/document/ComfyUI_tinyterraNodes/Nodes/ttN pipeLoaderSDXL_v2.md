---
tags:
- Image
- Pipeline
---

# pipeLoaderSDXL
## Documentation
- Class name: `ttN pipeLoaderSDXL_v2`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

This node is designed for advanced data loading and preprocessing in a pipeline, specifically tailored for handling large-scale datasets with efficiency and speed. It leverages specialized techniques to optimize data throughput and preprocessing, making it suitable for high-performance computing environments.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for loading the model, serving as a key input for initializing the pipeline with the desired model state.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`config_name`**
    - Determines the configuration to be used alongside the checkpoint, allowing for customization of the model's initialization parameters.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`vae_name`**
    - Identifies the VAE model to be loaded, crucial for the pipeline's operation in handling specific data transformations or enhancements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`clip_skip`**
    - Defines the number of layers to skip in the CLIP model, adjusting the depth of feature extraction and influencing the model's performance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`loras`**
    - Specifies custom LoRA configurations, enabling fine-tuning of the model's behavior through weight adjustments and optional CLIP weight modifications.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`refiner_ckpt_name`**
    - Optional input for loading a secondary refinement model, enhancing the pipeline's output through additional processing steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`refiner_config_name`**
    - Selects the configuration for the refinement model, similar to the main model configuration, for tailored processing adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`positive_g`**
    - Global positive textual inputs that specify overarching inclusion criteria for the dataset, guiding the model's focus.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`positive_l`**
    - Local positive textual inputs that detail finer inclusion criteria, allowing for more granular control over the dataset's composition.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_g`**
    - Global negative textual inputs that outline broad exclusion criteria, specifying what data or parameters should be avoided.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_l`**
    - Local negative textual inputs that provide detailed exclusion criteria, offering precise control over what is omitted from the dataset.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`conditioning_aspect`**
    - Defines the aspect ratio for conditioning, influencing the shape and dimensions of the generated or processed data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`conditioning_width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`conditioning_height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`crop_width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`crop_height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`target_aspect`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`target_width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`target_height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`positive_ascore`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`negative_ascore`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`empty_latent_aspect`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`empty_latent_width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`empty_latent_height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`batch_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`model_override`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`clip_override`**
    - unknown
    - Comfy dtype: `CLIP`
    - Python dtype: `unknown`
- **`optional_lora_stack`**
    - unknown
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `unknown`
- **`optional_controlnet_stack`**
    - unknown
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `unknown`
- **`refiner_model_override`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`refiner_clip_override`**
    - unknown
    - Comfy dtype: `CLIP`
    - Python dtype: `unknown`
- **`prepend_positive_g`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`prepend_positive_l`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`prepend_negative_g`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`prepend_negative_l`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
## Output types
- **`sdxl_pipe`**
    - Comfy dtype: `PIPE_LINE_SDXL`
    - The comprehensive pipeline output, encapsulating the processed data and model states for further use.
    - Python dtype: `str`
- **`model`**
    - Comfy dtype: `MODEL`
    - The loaded model's state, ready for integration into the pipeline for data processing or generation tasks.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning output, representing the data or features to be emphasized in the pipeline's processing.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning output, indicating the data or features to be de-emphasized or excluded.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model loaded into the pipeline, crucial for specific data transformations or enhancements.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model integrated into the pipeline, essential for extracting or processing textual and visual features.
    - Python dtype: `str`
- **`refiner_model`**
    - Comfy dtype: `MODEL`
    - The refinement model loaded into the pipeline for enhanced processing capabilities.
    - Python dtype: `str`
- **`refiner_positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning output from the refinement model, further specifying inclusion criteria.
    - Python dtype: `str`
- **`refiner_negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning output from the refinement model, further specifying exclusion criteria.
    - Python dtype: `str`
- **`refiner_clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model integrated into the refinement process, enhancing the pipeline's ability to process textual and visual features.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation generated by the pipeline, serving as a foundational component for further data manipulation or generation.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - A seed value for ensuring reproducibility in the pipeline's operations, affecting randomness in data processing.
    - Python dtype: `int`
- **`width`**
    - Comfy dtype: `INT`
    - The width dimension for the output data, defining the horizontal size of generated images or processed data.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height dimension for the output data, defining the vertical size of generated images or processed data.
    - Python dtype: `int`
- **`pos_string`**
    - Comfy dtype: `STRING`
    - A textual representation of the positive conditioning, summarizing the inclusion criteria for the dataset.
    - Python dtype: `str`
- **`neg_string`**
    - Comfy dtype: `STRING`
    - A textual representation of the negative conditioning, summarizing the exclusion criteria for the dataset.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeLoaderSDXL_v2:
    version = '2.1.0'
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["width x height [custom]",
                         "1024 x 1024 [S] 1:1",

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
        relative_ratios = ["width x height [custom]",
                           "1x Empty Latent Aspect",
                           "2x Empty Latent Aspect",
                           "3x Empty Latent Aspect",
                           "4x Empty Latent Aspect",
                           "5x Empty Latent Aspect",
                           "6x Empty Latent ASpect",
                           "7x Empty Latent Aspect",
                           "8x Empty Latent Aspect",
                           ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                        "loras": ("STRING", {"placeholder": "Loras - <lora:loraName:weight:optClipWeight>", "multiline": True}),

                        "refiner_ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                        "refiner_config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),

                        "positive_g": ("STRING", {"placeholder": "Linguistic Positive (positive_g)","multiline": True, "dynamicPrompts": True}),
                        "positive_l": ("STRING", {"placeholder": "Supporting Terms (positive_l)", "multiline": True, "dynamicPrompts": True}),
                        "negative_g": ("STRING", {"placeholder": "negative_g", "multiline": True, "dynamicPrompts": True}),
                        "negative_l": ("STRING", {"placeholder": "negative_l", "multiline": True, "dynamicPrompts": True}),

                        "conditioning_aspect": (relative_ratios, {"default": "1x Empty Latent Aspect"}),
                        "conditioning_width": ("INT", {"default": 2048.0, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "conditioning_height": ("INT", {"default": 2048.0, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        
                        "crop_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                        "crop_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),

                        "target_aspect": (relative_ratios, {"default": "1x Empty Latent Aspect"}),
                        "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                        "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                        
                        "positive_ascore": ("INT", {"default": 6.0, "min": 0, "step": 0.1}),
                        "negative_ascore": ("INT", {"default": 2.0, "min": 0, "step": 0.1}),

                        "empty_latent_aspect": (aspect_ratios, {"default": "1024 x 1024 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },                
                "optional": {
                    "model_override": ("MODEL",),
                    "clip_override": ("CLIP",),
                    "optional_lora_stack": ("LORA_STACK",),
                    "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                    "refiner_model_override": ("MODEL",),
                    "refiner_clip_override": ("CLIP",),
                    "prepend_positive_g": ("STRING", {"default": None, "forceInput": True}),
                    "prepend_positive_l": ("STRING", {"default": None, "forceInput": True}),
                    "prepend_negative_g": ("STRING", {"default": None, "forceInput": True}),
                    "prepend_negative_l": ("STRING", {"default": None, "forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoaderSDXL_v2.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("PIPE_LINE_SDXL" ,"MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONDITIONING", "CONDITIONING", "CLIP", "LATENT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("sdxl_pipe","model", "positive", "negative", "vae", "clip", "refiner_model", "refiner_positive", "refiner_negative", "refiner_clip", "latent", "seed", "width", "height", "pos_string", "neg_string")


    FUNCTION = "sdxl_pipeloader"
    CATEGORY = "🌏 tinyterra/pipe"

    def sdxl_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip, loras,
                        refiner_ckpt_name, refiner_config_name,
                        conditioning_aspect, conditioning_width, conditioning_height, crop_width, crop_height, target_aspect, target_width, target_height,
                        positive_g, positive_l, negative_g, negative_l,
                        positive_ascore, negative_ascore,
                        empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size, seed,
                        model_override=None, clip_override=None, optional_lora_stack=None, optional_controlnet_stack=None,
                        refiner_model_override=None, refiner_clip_override=None,
                        prepend_positive_g=None, prepend_positive_l=None, prepend_negative_g=None, prepend_negative_l=None,
                        prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height)
        samples = {"samples":latent}

        loader.clear_cache(prompt)
        model, clip, vae = loader.load_main3(ckpt_name, config_name, vae_name, loras, clip_skip, model_override, clip_override, optional_lora_stack, my_unique_id)

        if refiner_ckpt_name not in ["None", None]:
            refiner_model, refiner_clip, refiner_vae = loader.load_main3(refiner_ckpt_name, refiner_config_name, vae_name, None, clip_skip, refiner_model_override, refiner_clip_override)
        else:
            refiner_model, refiner_clip, refiner_vae = None, None, None


        if empty_latent_aspect and empty_latent_aspect != "width x height [custom]":
            empty_latent_width, empty_latent_height = empty_latent_aspect.replace(' ', '').split('[')[0].split('x')

        if conditioning_aspect and conditioning_aspect != "width x height [custom]":
            conditioning_factor = conditioning_aspect.split('x')[0]
            conditioning_width = int(conditioning_factor) * int(empty_latent_width)
            conditioning_height = int(conditioning_factor) * int(empty_latent_height)

        if target_aspect and target_aspect != "width x height [custom]":
            target_factor = target_aspect.split('x')[0]
            target_width = int(target_factor) * int(empty_latent_width)
            target_height = int(target_factor) * int(empty_latent_height)


        positive_embedding, refiner_positive_embedding = loader.embedding_encodeXL(positive_g, clip, seed=seed, title='pipeLoaderSDXL Positive', my_unique_id=my_unique_id, prepend_text=prepend_positive_g, text2=positive_l, prepend_text2=prepend_positive_l, width=conditioning_width, height=conditioning_height, crop_width=crop_width, crop_height=crop_height, target_width=target_width, target_height=target_height, refiner_clip=refiner_clip, ascore=positive_ascore)
        negative_embedding, refiner_negative_embedding = loader.embedding_encodeXL(negative_g, clip, seed=seed, title='pipeLoaderSDXL Negative', my_unique_id=my_unique_id, prepend_text=prepend_negative_g, text2=negative_l, prepend_text2=prepend_negative_l, width=conditioning_width, height=conditioning_height, crop_width=crop_width, crop_height=crop_height, target_width=target_width, target_height=target_height, refiner_clip=refiner_clip, ascore=negative_ascore)


        if optional_controlnet_stack is not None:
            for cnt in optional_controlnet_stack:
                positive_embedding, negative_embedding = loader.load_controlNet(positive_embedding, negative_embedding, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4])

        image = None

        sdxl_pipe = {"model": model,
                    "positive": positive_embedding,
                    "negative": negative_embedding,
                    "vae": vae,
                    "clip": clip,

                    "refiner_model": refiner_model,
                    "refiner_positive": refiner_positive_embedding,
                    "refiner_negative": refiner_negative_embedding,
                    "refiner_clip": refiner_clip,

                    "samples": samples,
                    "images": image,
                    "seed": seed,

                "loader_settings": None
        }

        final_positive = (prepend_positive_g + ' ' if prepend_positive_g else '') + (positive_g + ' ' if positive_g else '') + (prepend_positive_l + ' ' if prepend_positive_l else '') + (positive_l + ' ' if positive_l else '')
        final_negative = (prepend_negative_g + ' ' if prepend_negative_g else '') + (negative_g + ' ' if negative_g else '') + (prepend_negative_l + ' ' if prepend_negative_l else '') + (negative_l + ' ' if negative_l else '')

        return (sdxl_pipe, model, positive_embedding, negative_embedding, vae, clip, refiner_model, refiner_positive_embedding, refiner_negative_embedding, refiner_clip, samples, seed, empty_latent_width, empty_latent_height, final_positive, final_negative)

```
