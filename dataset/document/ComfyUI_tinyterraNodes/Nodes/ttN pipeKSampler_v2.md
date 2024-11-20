---
tags:
- Sampling
---

# pipeKSampler
## Documentation
- Class name: `ttN pipeKSampler_v2`
- Category: `🌏 tinyterra/pipe`
- Output node: `True`

The `ttN_pipeKSampler_v2` node is designed for advanced image sampling and processing within a pipeline. It integrates various functionalities such as LoRA adjustments, noise management, step configuration, and optional model enhancements to facilitate customized image generation and manipulation. This node emphasizes flexibility in image output options and the ability to refine results through denoising and upscale methods, catering to diverse requirements in image synthesis workflows.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration and state, serving as the foundation for the sampling process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`lora_name`**
    - Specifies the LoRA model to be applied for adjustments, enhancing the control over model behavior and output quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_strength`**
    - Determines the intensity of the LoRA adjustments, allowing for fine-tuning of the model's influence on the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`upscale_method`**
    - Defines the method used for upscaling images, offering options to enhance image resolution.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_model_name`**
    - Specifies the model used for upscaling, enabling high-quality image enlargement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - Sets the scaling factor for upscaling, determining the increase in image size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rescale`**
    - Controls how images are rescaled, providing various options for adjusting image dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`percent`**
    - Determines the percentage for image rescaling, allowing for flexible size adjustment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`width`**
    - Specifies the target width for image resizing, enabling precise dimension control.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the target height for image resizing, facilitating specific size requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`longer_side`**
    - Defines the target size for the longer side of the image, ensuring aspect ratio maintenance during resizing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop`**
    - Determines the cropping method, offering strategies for image composition adjustment.
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
    - Controls the denoising level applied to images, affecting clarity and detail preservation.
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
    - Sets the seed for random number generation, ensuring reproducibility in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_model`**
    - Allows for the specification of an alternative model for sampling, providing customization options.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`optional_positive`**
    - Specifies optional positive conditioning to influence the image generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_negative`**
    - Specifies optional negative conditioning to steer the image generation away from certain attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_latent`**
    - Allows for the input of a latent image representation, offering advanced customization in the generation process.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`optional_vae`**
    - Specifies an optional VAE model for use in the image generation process, enabling further customization.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`optional_clip`**
    - Allows for the specification of an optional CLIP model, providing additional control over image relevance to text descriptions.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`input_image_override`**
    - Enables the overriding of the input image, allowing for direct manipulation of the image generation process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`adv_xyPlot`**
    - Specifies an advanced XY plot for visualization or analysis, integrating additional data dimensions into the process.
    - Comfy dtype: `ADV_XYPLOT`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the updated pipeline configuration after processing, reflecting changes made during sampling.
    - Python dtype: `dict`
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the model used in the sampling process, including any adjustments or selections made.
    - Python dtype: `str`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive conditioning applied, influencing the generation towards desired outcomes.
    - Python dtype: `str`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative conditioning applied, steering the generation away from certain attributes.
    - Python dtype: `str`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Returns the latent representation of the generated image, providing a basis for further manipulation or analysis.
    - Python dtype: `str`
- **`vae`**
    - Comfy dtype: `VAE`
    - Outputs the VAE model used in the process, if any, highlighting its role in image reconstruction.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the CLIP model applied, if any, underscoring its influence on aligning images with textual descriptions.
    - Python dtype: `str`
- **`images`**
    - Comfy dtype: `IMAGE`
    - Outputs the final generated images, showcasing the results of the sampling and processing.
    - Python dtype: `str`
- **`seed`**
    - Comfy dtype: `INT`
    - Returns the seed value used in the process, ensuring traceability of the generation.
    - Python dtype: `int`
- **`plot_image`**
    - Comfy dtype: `IMAGE`
    - Provides a visual representation or plot related to the sampling process, offering insights into the generation.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeKSampler_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),

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

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                 "ttNnodeVersion": ttN_pipeKSampler_v2.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def sample(self, pipe,
               lora_name, lora_strength,
               steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None,
               seed=None, adv_xyPlot=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
               prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        samp_model = optional_model if optional_model is not None else pipe["model"]
        samp_positive = optional_positive if optional_positive is not None else pipe["positive"]
        samp_negative = optional_negative if optional_negative is not None else pipe["negative"]
        samp_samples = optional_latent if optional_latent is not None else pipe["samples"]
        samp_images = input_image_override if input_image_override is not None else pipe["images"]
        samp_vae = optional_vae if optional_vae is not None else pipe["vae"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        if seed in (None, 'undefined'):
            samp_seed = pipe["seed"]
        else:
            samp_seed = seed
            
        del pipe
        
        def process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                samp_model, samp_clip = loader.load_lora(lora_name, samp_model, samp_clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            if upscale_method != "None":
                samp_samples = sampler.handle_upscale(samp_samples, upscale_method, factor, crop, upscale_model_name, samp_vae, samp_images, rescale, percent, width, height, longer_side)

            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      
            results = list()
            if (image_output != "Disabled"):
                # Save images
                latent = samp_samples["samples"]
                samp_images = samp_vae.decode(latent)

                results = ttN_save.images(samp_images, save_prefix, image_output, embed_workflow, file_type)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": samp_images,
                "seed": samp_seed,

                "loader_settings": None,
            }

            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output(new_pipe), None)

            return {"ui": {"images": results},
                    "result": (*sampler.get_output(new_pipe), None)}

        def process_xyPlot(samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)

            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samples,
                "images": images,
                "seed": samp_seed,

                "loader_settings": None,
            }

            if image_output in ("Hide", "Hide/Save"):
                return (*sampler.get_output(new_pipe), plot_image)

            return {"ui": {"images": plot_result}, "result": (*sampler.get_output(new_pipe), plot_image)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_strength, lora_strength,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent)
        else:
            return process_xyPlot(samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_strength, lora_strength, steps, cfg, sampler_name, 
                                  scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

```
