# EasyLoader (PixArt)
## Documentation
- Class name: `easy pixArtLoader`
- Category: `EasyUse/Loaders`
- Output node: `False`

The pixArtLoader node is designed to facilitate the loading and integration of PixArt models into a broader AI-driven art generation framework. It abstracts the complexities involved in initializing and configuring PixArt models, making it easier for users to leverage these models for creative and generative tasks.
## Input types
### Required
- **`ckpt_name`**
    - Specifies the checkpoint name for the PixArt model to be loaded. This parameter is crucial for identifying the specific model version and its associated weights for initialization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`model_name`**
    - Determines the PixArt model to be used for the art generation process. This selection influences the style and characteristics of the generated artwork.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`vae_name`**
    - Identifies the VAE (Variational Autoencoder) model associated with the PixArt model. This is important for the preprocessing and postprocessing stages of the art generation pipeline.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`t5_type`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`clip_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`padding`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`t5_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`device`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`dtype`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`lora_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`lora_model_strength`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`ratio`**
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
- **`positive`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`negative`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`batch_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`optional_lora_stack`**
    - unknown
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `unknown`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the pipeline component configured with the PixArt model for art generation.
    - Python dtype: `torch.nn.Module`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the loaded PixArt model instance.
    - Python dtype: `torch.nn.Module`
- **`vae`**
    - Comfy dtype: `VAE`
    - Returns the VAE model associated with the PixArt model, used in the art generation process.
    - Python dtype: `torch.nn.Module`
- **`ui`**
    - Provides a user interface component for interacting with the PixArt model configurations and settings.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class pixArtLoader:
    @classmethod
    def INPUT_TYPES(cls):
        for k in range(1, torch.cuda.device_count()):
            devices.append(f"cuda:{k}")
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                "model_name":(list(pixart_conf.keys()),),
                "vae_name": (folder_paths.get_filename_list("vae"),),
                "t5_type": (['sd3'],),
                "clip_name": (folder_paths.get_filename_list("clip"),),
                "padding": ("INT", {"default": 1, "min": 1, "max": 300}),
                "t5_name": (folder_paths.get_filename_list("t5"),),
                "device": (devices, {"default": "cpu"}),
                "dtype": (dtypes,),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "ratio": (["custom"] + list(pixart_res["PixArtMS_XL_2"].keys()), {"default":"1.00"}),
                "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),

                "positive": ("STRING", {"default": "", "placeholder": "Positive", "multiline": True}),
                "negative": ("STRING", {"default": "", "placeholder": "Negative", "multiline": True}),

                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional":{
              "optional_lora_stack": ("LORA_STACK",),
            },
            "hidden": {"prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "VAE")
    RETURN_NAMES = ("pipe", "model", "vae")
    FUNCTION = "pixart_pipeloader"
    CATEGORY = "EasyUse/Loaders"

    def pixart_pipeloader(self, ckpt_name, model_name, vae_name, t5_type, clip_name, padding, t5_name, device, dtype, lora_name, lora_model_strength, ratio, empty_latent_width, empty_latent_height, positive, negative, batch_size, optional_lora_stack=None, prompt=None, my_unique_id=None):
        # Clean models from loaded_objects
        easyCache.update_loaded_objects(prompt)

        # load checkpoint
        model = easyCache.load_dit_ckpt(ckpt_name=ckpt_name, model_name=model_name, pixart_conf=pixart_conf,
                                        model_type='PixArt')
        # load vae
        vae = easyCache.load_vae(vae_name)

        # load t5
        if t5_type == 'sd3':
            clip = easyCache.load_clip(clip_name=clip_name,type='sd3')
            clip = easyCache.load_t5_from_sd3_clip(sd3_clip=clip, padding=padding)
            lora_stack = None
            if optional_lora_stack is not None:
                for lora in optional_lora_stack:
                    lora = {"lora_name": lora[0], "model": model, "clip": clip, "model_strength": lora[1],
                            "clip_strength": lora[2]}
                    model, _ = easyCache.load_lora(lora, type='PixArt')
                    lora['model'] = model
                    lora['clip'] = clip
                    lora_stack.append(lora)

            if lora_name != "None":
                lora = {"lora_name": lora_name, "model": model, "clip": clip, "model_strength": lora_model_strength,
                        "clip_strength": 1}
                model, _ = easyCache.load_lora(lora, type='PixArt')
                lora_stack.append(lora)

            positive_embeddings_final, = CLIPTextEncode().encode(clip, positive)
            negative_embeddings_final, = CLIPTextEncode().encode(clip, negative)
        else:
            # todo t5v11
            positive_embeddings_final, negative_embeddings_final = None, None
            clip = None
            pass

        # Create Empty Latent
        if ratio != 'custom':
            if model_name in ['ControlPixArtMSHalf','PixArtMS_Sigma_XL_2_900M']:
                res_name = 'PixArtMS_XL_2'
            elif model_name in ['ControlPixArtHalf']:
                res_name = 'PixArt_XL_2'
            else:
                res_name = model_name
            width, height = pixart_res[res_name][ratio]
            empty_latent_width = width
            empty_latent_height = height

        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8], device=sampler.device)
        samples = {"samples": latent}

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
                "t5_name": t5_name,

                "positive": positive,
                "negative": negative,
                "ratio": ratio,
                "empty_latent_width": empty_latent_width,
                "empty_latent_height": empty_latent_height,
                "batch_size": batch_size,
            }
        }

        return {"ui": {},
                "result": (pipe, model, vae, clip, positive_embeddings_final, negative_embeddings_final, samples)}

```
