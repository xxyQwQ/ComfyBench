---
tags:
- Face
---

# EasyLoader (HunYuanDiT)
## Documentation
- Class name: `easy hunyuanDiTLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

This node is designed to load and configure the HunYuanDiT model for image generation tasks. It prepares the model by loading the necessary checkpoints, VAE, and CLIP components, and sets up the environment for generating images based on text prompts. The node facilitates the integration of the HunYuanDiT model into the workflow, ensuring that all components are correctly initialized and ready for use.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for loading the model, essential for initializing the HunYuanDiT model with the correct weights and configurations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model_name`**
    - Defines the specific model configuration to use, allowing for customization and optimization of the image generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`vae_name`**
    - Indicates the VAE component to load, crucial for the image encoding and decoding processes within the HunYuanDiT model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_name`**
    - Determines the CLIP model to integrate, used for processing text prompts and guiding the image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mt5_name`**
    - Specifies the MT5 model for additional text processing capabilities, enhancing the model's understanding of text prompts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`device`**
    - Selects the computation device (e.g., CPU, GPU) for model execution, affecting performance and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`dtype`**
    - Sets the data type for model operations, ensuring compatibility and optimal performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`resolution`**
    - Defines the resolution of the generated images, allowing for control over image quality and detail.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`empty_latent_width`**
    - Specifies the width of the empty latent space, important for initializing the image generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`empty_latent_height`**
    - Specifies the height of the empty latent space, important for initializing the image generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`positive`**
    - Contains positive text prompts that guide the image generation towards desired attributes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative`**
    - Contains negative text prompts that guide the image generation away from undesired attributes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`batch_size`**
    - Determines the number of images to generate in a single batch, impacting performance and output volume.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the configured pipeline for image generation, including all necessary models and settings.
    - Python dtype: `object`
- **`model`**
    - Comfy dtype: `MODEL`
    - Provides the loaded and configured model ready for image generation tasks.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the loaded VAE component, essential for the image encoding and decoding processes.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class hunyuanDiTLoader:
    @classmethod
    def INPUT_TYPES(cls):
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_name":(list(hydit_conf.keys()),{"default":"G/2"}),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "mt5_name": (folder_paths.get_filename_list("t5"),),
                "device": (devices, {"default": "cpu"}),
                "dtype": (dtypes,),
                "resolution": (resolution_strings,{'default':'1024 x 1024'}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")
    FUNCTION = "hydit_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def text_encoder(self, text, text_t5, clip, t5):
        # T5
        t5.load_model()
        t5_pre = t5.tokenizer(
            text_t5,
            max_length=t5.cond_stage_model.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        t5_mask = t5_pre["attention_mask"]
        with torch.no_grad():
            t5_outs = t5.cond_stage_model.transformer(
                input_ids=t5_pre["input_ids"].to(t5.load_device),
                attention_mask=t5_mask.to(t5.load_device),
                output_hidden_states=True,
            )
            # to-do: replace -1 for clip skip
            t5_embs = t5_outs["hidden_states"][-1].float().cpu()

        # clip
        clip.load_model()
        clip_pre = clip.tokenizer(
            text,
            max_length=clip.cond_stage_model.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        clip_mask = clip_pre["attention_mask"]
        with torch.no_grad():
            clip_outs = clip.cond_stage_model.transformer(
                input_ids=clip_pre["input_ids"].to(clip.load_device),
                attention_mask=clip_mask.to(clip.load_device),
            )
            # to-do: add hidden states
            clip_embs = clip_outs[0].float().cpu()

        # combined cond
        return ([[
            clip_embs, {
                "context_t5": t5_embs,
                "context_mask": clip_mask.float(),
                "context_t5_mask": t5_mask.float()
            }
        ]],)

    def hydit_pipeloader(self, ckpt_name, model_name, vae_name, clip_name, mt5_name, device, dtype, resolution, empty_latent_width, empty_latent_height, positive, negative, batch_size, prompt=None, my_unique_id=None):
        dtype = string_to_dtype(dtype, "text_encoder")
        if device == "cpu":
            assert dtype in [None, torch.float32,
                             torch.bfloat16], f"Can't use dtype '{dtype}' with CPU! Set dtype to 'default' or 'bf16'."

        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # Load models
        log_node_warn("正在加载模型...")
        # load checkpoint
        model = easyCache.load_dit_ckpt(ckpt_name=ckpt_name, model_name=model_name, hydit_conf=hydit_conf, model_type='HyDiT')
        # load vae
        vae = easyCache.load_vae(vae_name)
        # load clip
        clip = easyCache.load_dit_clip(clip_name=clip_name, device=device, dtype=dtype, model_type='HyDiT')
        # load t5
        t5 = easyCache.load_dit_t5(t5_name=mt5_name, device=device, dtype=dtype, model_type='HyDiT')

        # Create Empty Latent
        samples = sampler.emptyLatent(resolution, empty_latent_width, empty_latent_height, batch_size, sd3=False)

        # t5 text encoder
        positive_embeddings_final, = self.text_encoder(positive, positive, clip, t5)
        negative_embeddings_final, = self.text_encoder(negative, negative, clip, t5)

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
                "clip_name": clip_name,
                "vae_name": vae_name,
                "mt5_name": mt5_name,

                "positive": positive,
                "negative": negative,
                "resolution": resolution,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {},
                "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}

```
