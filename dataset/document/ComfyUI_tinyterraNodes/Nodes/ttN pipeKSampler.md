---
tags:
- Sampling
---

# pipeKSampler v1 (Legacy)
## Documentation
- Class name: `ttN pipeKSampler`
- Category: `🌏 tinyterra/legacy`
- Output node: `True`

The ttN_pipeKSampler node is designed for sampling and processing images within a pipeline, incorporating various enhancements and adjustments such as noise addition or reduction, upscaling, and applying LoRA adjustments. It integrates with other components to manage image generation, modification, and output based on specified configurations, aiming to produce high-quality images tailored to user inputs and preferences.
## Input types
### Required
- **`pipe`**
    - Represents the current state of the pipeline, including models, embeddings, and other relevant settings, serving as the foundation for the sampling process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`lora_name`**
    - Specifies the LoRA model to be applied for adjustments, enhancing the sampling process with model-specific enhancements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_model_strength`**
    - Determines the intensity of the LoRA model adjustments, affecting the final image quality and characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_clip_strength`**
    - Specifies the intensity of LoRA adjustments on the CLIP model, influencing the relevance of text-image alignment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`upscale_method`**
    - Defines the method used for upscaling the images, enhancing the resolution and clarity.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - The factor by which the image is upscaled, determining the increase in resolution.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`crop`**
    - Describes the method used for cropping the images, affecting the final composition and focus.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sampler_state`**
    - Indicates the state of the sampler, whether to sample new images or hold the current state.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`cfg`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sampler_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`scheduler`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`denoise`**
    - The level of denoising applied to the images, affecting the smoothness and clarity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_output`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`save_prefix`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
### Optional
- **`seed`**
    - Seed value for random number generation, ensuring reproducibility and consistency in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_model`**
    - An optional model parameter for further customization of the sampling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`optional_positive`**
    - Optional positive conditioning to guide the image generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_negative`**
    - Optional negative conditioning to steer the image generation away from certain attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_latent`**
    - Optional latent image input for initializing or influencing the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`optional_vae`**
    - An optional VAE model parameter for decoding or encoding latent images.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`optional_clip`**
    - An optional CLIP model parameter for text-image alignment enhancements.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`xyPlot`**
    - Optional XY plot data for guiding the sampling process with specific patterns or distributions.
    - Comfy dtype: `XYPLOT`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the updated pipeline state, including models, embeddings, and settings after processing.
    - Python dtype: `dict`
- **`model`**
    - Comfy dtype: `MODEL`
    - The model used in the sampling process, potentially adjusted by LoRA or other modifications.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Positive conditioning applied during the sampling process, guiding towards desired attributes.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Negative conditioning applied during the sampling process, steering away from certain attributes.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation of the sampled image, suitable for further processing or analysis.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - The VAE model used for decoding the latent image, if applicable.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model used for text-image alignment enhancements, if applicable.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final output image after sampling and processing.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - The seed value used during the sampling process, ensuring reproducibility.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ImageUpscaleWithModel](../../Comfy/Nodes/ImageUpscaleWithModel.md)



## Source code
```python
class ttN_TSC_pipeKSampler:
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
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
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
                 "ttNnodeVersion": ttN_TSC_pipeKSampler.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/legacy"

    def sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, seed=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        samp_model = optional_model if optional_model is not None else pipe["model"]
        samp_positive = optional_positive if optional_positive is not None else pipe["positive"]
        samp_negative = optional_negative if optional_negative is not None else pipe["negative"]
        samp_samples = optional_latent if optional_latent is not None else pipe["samples"]
        samp_vae = optional_vae if optional_vae is not None else pipe["vae"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        if seed in (None, 'undefined'):
            samp_seed = pipe["seed"]
        else:
            samp_seed = seed      

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                samp_model, samp_clip = loader.load_lora(lora_name, samp_model, samp_clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            samp_samples = sampler.handle_upscale(samp_samples, upscale_method, factor, crop)

            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      

            latent = samp_samples["samples"]
            samp_images = samp_vae.decode(latent)

            results = ttN_save.images(samp_images, save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": samp_images,
                "seed": samp_seed,

                "loader_settings": pipe["loader_settings"],
            }
            
            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)
            
            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe)}

        def process_hold_state(pipe, image_output, my_unique_id):
            last_pipe = sampler.init_state(my_unique_id, "pipe_line", pipe)

            last_results = sampler.init_state(my_unique_id, "results", list())
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(last_pipe)

            return {"ui": {"images": last_results}, "result": sampler.get_output(last_pipe)} 

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot):
            
            random.seed(seed)
            
            sampleXYplot = ttNxyPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": lora_name, "lora_model_strength": lora_model_strength, "lora_clip_strength": lora_clip_strength,
                "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "seed": samp_seed,

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive, "negative_cond": samp_negative,
                
                "ckpt_name": pipe['loader_settings']['ckpt_name'],
                "vae_name": pipe['loader_settings']['vae_name'],
                "clip_skip": pipe['loader_settings']['clip_skip'],
                "lora1_name": pipe['loader_settings']['lora1_name'],
                "lora1_model_strength": pipe['loader_settings']['lora1_model_strength'],
                "lora1_clip_strength": pipe['loader_settings']['lora1_clip_strength'],
                "lora2_name": pipe['loader_settings']['lora2_name'],
                "lora2_model_strength": pipe['loader_settings']['lora2_model_strength'],
                "lora2_clip_strength": pipe['loader_settings']['lora2_clip_strength'],
                "lora3_name": pipe['loader_settings']['lora3_name'],
                "lora3_model_strength": pipe['loader_settings']['lora3_model_strength'],
                "lora3_clip_strength": pipe['loader_settings']['lora3_clip_strength'],
                "positive": pipe['loader_settings']['positive'],
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'],
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'],
                "negative": pipe['loader_settings']['negative'],
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'],
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'],
                }
            
            latent_image = sampleXYplot.get_latent(pipe["samples"])
            
            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step, last_step, force_full_denoise, disable_noise)

            samp_samples = {"samples": latents_plot}
            images = sampleXYplot.plot_images_and_labels()

            if xyPlot["output_individuals"]:
                results = ttN_save.images(images, save_prefix, image_output)
            else:
                results = ttN_save.images(images[-1], save_prefix, image_output)
                

            sampler.update_value_by_id("results", my_unique_id, results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": images,
                "seed": samp_seed,

                "loader_settings": pipe["loader_settings"],
            }

            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": sampler.get_output(new_pipe)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        if sampler_state == "Sample" and xyPlot is None:
            return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

        elif sampler_state == "Sample" and xyPlot is not None:
            return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(pipe, image_output, my_unique_id)

```
