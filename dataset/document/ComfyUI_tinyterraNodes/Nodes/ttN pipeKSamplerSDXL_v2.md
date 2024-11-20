---
tags:
- Sampling
---

# pipeKSamplerSDXL
## Documentation
- Class name: `ttN pipeKSamplerSDXL_v2`
- Category: `🌏 tinyterra/pipe`
- Output node: `True`

This node is designed to facilitate the sampling process within a specific pipeline, focusing on handling and transforming data through various stages of sampling, including initialization, processing, and output generation. It abstracts the complexities involved in managing state, handling image outputs, and integrating with other components like models and samplers, aiming to streamline the workflow for generating and refining samples based on given inputs.
## Input types
### Required
- **`sdxl_pipe`**
    - Represents the pipeline state or configuration that is being passed through the sampling process. It is crucial for maintaining continuity and consistency across different stages of sampling.
    - Comfy dtype: `PIPE_LINE_SDXL`
    - Python dtype: `Dict`
- **`lora_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`lora_strength`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`upscale_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`upscale_model_name`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`factor`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`rescale`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`percent`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`width`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`height`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`longer_side`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`crop`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`base_steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`cfg`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`denoise`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`refiner_steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`refiner_cfg`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`refiner_denoise`**
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
- **`image_output`**
    - Specifies the desired output handling for images, such as whether to hide or save them, affecting how the node processes and returns image data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`file_type`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`embed_workflow`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
### Optional
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`optional_model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`optional_positive`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`optional_negative`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`optional_latent`**
    - unknown
    - Comfy dtype: `LATENT`
    - Python dtype: `unknown`
- **`optional_vae`**
    - unknown
    - Comfy dtype: `VAE`
    - Python dtype: `unknown`
- **`optional_refiner_model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`optional_refiner_positive`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`optional_refiner_negative`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`optional_clip`**
    - unknown
    - Comfy dtype: `CLIP`
    - Python dtype: `unknown`
- **`input_image_override`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`adv_xyPlot`**
    - unknown
    - Comfy dtype: `ADV_XYPLOT`
    - Python dtype: `unknown`
## Output types
- **`sdxl_pipe`**
    - Comfy dtype: `PIPE_LINE_SDXL`
    - unknown
    - Python dtype: `unknown`
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - unknown
    - Python dtype: `unknown`
- **`model`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - unknown
    - Python dtype: `unknown`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - unknown
    - Python dtype: `unknown`
- **`refiner_model`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
- **`refiner_positive`**
    - Comfy dtype: `CONDITIONING`
    - unknown
    - Python dtype: `unknown`
- **`refiner_negative`**
    - Comfy dtype: `CONDITIONING`
    - unknown
    - Python dtype: `unknown`
- **`latent`**
    - Comfy dtype: `LATENT`
    - unknown
    - Python dtype: `unknown`
- **`vae`**
    - Comfy dtype: `VAE`
    - unknown
    - Python dtype: `unknown`
- **`clip`**
    - Comfy dtype: `CLIP`
    - unknown
    - Python dtype: `unknown`
- **`images`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`seed`**
    - Comfy dtype: `INT`
    - unknown
    - Python dtype: `unknown`
- **`plot_image`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`ui`**
    - Contains the UI elements or data to be displayed, typically including images or results from the sampling process.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeKSamplerSDXL_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"sdxl_pipe": ("PIPE_LINE_SDXL",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (UPSCALE_METHODS, {"default": "None"}),
                "upscale_model_name": (UPSCALE_MODELS,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect', 'None'],),
                "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "crop": (CROP_METHODS,),

                "base_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                "refiner_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "refiner_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES, {"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_refiner_model": ("MODEL",),
                "optional_refiner_positive": ("CONDITIONING",),
                "optional_refiner_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_pipeKSamplerSDXL_v2.version},
        }

    RETURN_TYPES = ("PIPE_LINE_SDXL", "PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("sdxl_pipe", "pipe","model", "positive", "negative" , "refiner_model", "refiner_positive", "refiner_negative", "latent", "vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def sample(self, sdxl_pipe,
               lora_name, lora_strength,
               base_steps, refiner_steps, cfg, denoise, refiner_cfg, refiner_denoise, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow,
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None, adv_xyPlot=None,
               seed=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
               prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False,
               optional_refiner_model=None, optional_refiner_positive=None, optional_refiner_negative=None):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        sdxl_model = optional_model if optional_model is not None else sdxl_pipe["model"]
        sdxl_positive = optional_positive if optional_positive is not None else sdxl_pipe["positive"]
        sdxl_negative = optional_negative if optional_negative is not None else sdxl_pipe["negative"]
        sdxl_samples = optional_latent if optional_latent is not None else sdxl_pipe["samples"]
        sdxl_images = input_image_override if input_image_override is not None else sdxl_pipe["images"]
        sdxl_vae = optional_vae if optional_vae is not None else sdxl_pipe["vae"]
        sdxl_clip = optional_clip if optional_clip is not None else sdxl_pipe["clip"]

        sdxl_refiner_model = optional_refiner_model if optional_refiner_model is not None else sdxl_pipe["refiner_model"]
        sdxl_refiner_positive = optional_refiner_positive if optional_refiner_positive is not None else sdxl_pipe["refiner_positive"]
        #sdxl_refiner_positive = sdxl_positive if sdxl_refiner_positive is None else sdxl_refiner_positive
        sdxl_refiner_negative = optional_refiner_negative if optional_refiner_negative is not None else sdxl_pipe["refiner_negative"]
        #sdxl_refiner_negative = sdxl_negative if sdxl_refiner_negative is None else sdxl_refiner_negative
        sdxl_refiner_clip = sdxl_pipe["refiner_clip"]

        if seed in (None, 'undefined'):
            sdxl_seed = sdxl_pipe["seed"]
        else:
            sdxl_seed = seed    
  
        del sdxl_pipe
        
        def process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, my_unique_id, preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            
            # Load Lora
            if lora_name not in (None, "None"):
                sdxl_model, sdxl_clip = loader.load_lora(lora_name, sdxl_model, sdxl_clip, lora_model_strength, lora_clip_strength)
            
            total_steps = base_steps + refiner_steps

            # Upscale samples if enabled
            if upscale_method != "None":
                sdxl_samples = sampler.handle_upscale(sdxl_samples, upscale_method, factor, crop, upscale_model_name, sdxl_vae, sdxl_images, rescale, percent, width, height, longer_side,)

            if (refiner_steps > 0) and (sdxl_refiner_model not in [None, "None"]):
                # Base Sample
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, total_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=force_full_denoise, disable_noise=disable_noise)

                # Refiner Sample
                sdxl_samples = sampler.common_ksampler(sdxl_refiner_model, sdxl_seed, total_steps, refiner_cfg, sampler_name, scheduler, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_samples,
                                                       denoise=refiner_denoise, preview_latent=preview_latent, start_step=base_steps, last_step=10000, force_full_denoise=True, disable_noise=True)
            else:
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, base_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=True, disable_noise=disable_noise)

            results = list()
            if (image_output != "Disabled"):
                latent = sdxl_samples["samples"]
                sdxl_images = sdxl_vae.decode(latent)

                results = ttN_save.images(sdxl_images, save_prefix, image_output, embed_workflow, file_type)

            new_sdxl_pipe = {
                "model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "refiner_model": sdxl_refiner_model,
                "refiner_positive": sdxl_refiner_positive,
                "refiner_negative": sdxl_refiner_negative,
                "refiner_clip": sdxl_refiner_clip,

                "samples": sdxl_samples,
                "images": sdxl_images,
                "seed": sdxl_seed,

                "loader_settings": None,
            }
            
            pipe = {"model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "samples": sdxl_samples,
                "images": sdxl_images,
                "seed": sdxl_seed,
                
                "loader_settings": None,  
            }
            
            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), None)

            return {"ui": {"images": results},
                    "result": (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), None)}

        def process_xyPlot(sdxl_model, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                           base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)
            
            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise,
                                 image_output, save_prefix, prompt, my_unique_id, preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            new_sdxl_pipe = {
                "model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "refiner_model": sdxl_refiner_model,
                "refiner_positive": sdxl_refiner_positive,
                "refiner_negative": sdxl_refiner_negative,
                "refiner_clip": sdxl_refiner_clip,

                "samples": samples,
                "images": images,
                "seed": sdxl_seed,

                "loader_settings": None,
            }
            
            pipe = {"model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "samples": samples,
                "images": images,
                "seed": sdxl_seed,
                
                "loader_settings": None,  
            }

            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), plot_image)

            return {"ui": {"images": plot_result},
                    "result": (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), plot_image)}
            
        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative,
                                        lora_name, lora_strength, lora_strength,
                                        sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise, image_output, save_prefix, file_type, embed_workflow, prompt, my_unique_id, preview_latent)
        else:
            return process_xyPlot(sdxl_model, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_strength, lora_strength,
                           base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

```
