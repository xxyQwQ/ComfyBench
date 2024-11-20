---
tags:
- Sampling
---

# pipeKSamplerAdvanced v1 (Legacy)
## Documentation
- Class name: `ttN pipeKSamplerAdvanced`
- Category: `🌏 tinyterra/legacy`
- Output node: `True`

This node is designed for advanced sampling in generative models, incorporating various techniques such as LoRA adjustments, noise control, and optional model components to refine and generate high-quality samples. It supports complex workflows including image upscaling, embedding workflows, and handling of optional inputs for enhanced flexibility and customization in sample generation.
## Input types
### Required
- **`pipe`**
    - unknown
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `unknown`
- **`lora_name`**
    - Specifies the name of the LoRA model to be used for adjustments, enhancing the control over the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_model_strength`**
    - Determines the strength of the LoRA model adjustments, allowing for fine-tuning of the model's behavior during sampling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_clip_strength`**
    - Specifies the strength of the LoRA clip adjustments, impacting the final visual quality of the samples.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`upscale_method`**
    - Specifies the method used for upscaling the generated samples, affecting the resolution and clarity.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - The factor by which the image is upscaled, directly influencing the output image size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`crop`**
    - Determines if and how the output images are cropped, affecting the final composition and aspect ratio.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sampler_state`**
    - Represents the state of the sampler, including configurations and parameters for the sampling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`add_noise`**
    - Controls whether noise is added to the sampling process, with options to enable or disable noise, affecting the texture and details of the generated samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`steps`**
    - Defines the number of steps to perform in the sampling process, directly impacting the quality and characteristics of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration setting for the sampling process, providing a means to customize various aspects of sample generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampler to be used, allowing for selection among multiple sampling strategies.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Specifies the scheduler for controlling the sampling process, aiding in the management of sampling steps and their execution.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at_step`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`end_at_step`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`return_with_leftover_noise`**
    - Controls whether the final output includes leftover noise, affecting the visual texture of the samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`image_output`**
    - Determines how the output images are handled, including options for hiding or saving the generated samples.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - Prefix for saving the generated samples, facilitating organization and retrieval of output files.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`noise_seed`**
    - The seed used for generating noise, ensuring reproducibility and consistency in the noise added to the samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_model`**
    - Specifies an optional model to be used in the sampling process, allowing for customization and experimentation.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`optional_positive`**
    - Optional positive conditioning to guide the sampling towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_negative`**
    - Optional negative conditioning to steer the sampling away from undesired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_latent`**
    - Specifies an optional latent input for the sampling process, providing a starting point or influence.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`optional_vae`**
    - Specifies an optional VAE model to be used, affecting the encoding and decoding of samples.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`optional_clip`**
    - Specifies an optional CLIP model for additional guidance or conditioning in the sampling process.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`xyPlot`**
    - Optional XY plot data for visualization or analysis purposes during the sampling process.
    - Comfy dtype: `XYPLOT`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The pipeline configuration used for the sampling process.
    - Python dtype: `str`
- **`model`**
    - Comfy dtype: `MODEL`
    - The model used in the sampling process.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Positive conditioning data used to guide the sampling process.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Negative conditioning data used to steer the sampling away from undesired attributes.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation of the sample.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model used in the sampling process, if applicable.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model used for additional guidance or conditioning, if applicable.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final image output from the sampling process.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - The seed value used for reproducibility in the sampling process.
    - Python dtype: `int`
- **`ui`**
    - Provides a user interface component for visualizing the generated samples, enhancing the interaction and presentation of results.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeKSamplerAdvanced:
    version = '1.0.5'
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (cls.upscale_methods,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "crop": (cls.crop_methods,),
                "sampler_state": (["Sample", "Hold"], ),

                "add_noise": (["enable", "disable"], ),

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),

                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], ),

                "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"})
                },
                "optional": 
                {"noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "optional_model": ("MODEL",),
                 "optional_positive": ("CONDITIONING",),
                 "optional_negative": ("CONDITIONING",),
                 "optional_latent": ("LATENT",),
                 "optional_vae": ("VAE",),
                 "optional_clip": ("CLIP",),
                 "xyPlot": ("XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                 "embeddingsList": (folder_paths.get_filename_list("embeddings"),),
                 "ttNnodeVersion": ttN_pipeKSamplerAdvanced.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/legacy"

    def sample(self, pipe,
               lora_name, lora_model_strength, lora_clip_strength,
               sampler_state, add_noise, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               noise_seed=None, optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_at_step=None, end_at_step=None, return_with_leftover_noise=False):
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
            
        out = ttN_TSC_pipeKSampler.sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise, 
               optional_model, optional_positive, optional_negative, optional_latent, optional_vae, optional_clip, noise_seed, xyPlot, upscale_method, factor, crop, prompt, extra_pnginfo, my_unique_id, start_at_step, end_at_step, force_full_denoise, disable_noise)

        return out 

```
