---
tags:
- Sampling
---

# tinyKSampler
## Documentation
- Class name: `ttN KSampler_v2`
- Category: `🌏 tinyterra/base`
- Output node: `True`

The ttN KSampler_v2 node is designed for advanced sampling in generative models, offering a flexible interface to control the sampling process. It integrates various sampling and scheduling strategies, noise control mechanisms, and optional conditioning inputs to generate or refine images based on latent representations. This node is tailored for customization and experimentation in image generation workflows, enabling users to fine-tune the balance between fidelity and diversity in the generated outputs.
## Input types
### Required
- **`model`**
    - Specifies the generative model used for sampling, central to the generation process.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`positive`**
    - Optional positive conditioning to guide the generation towards desired attributes or content.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `object`
- **`negative`**
    - Optional negative conditioning to steer the generation away from undesired attributes or content.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `object`
- **`latent`**
    - unknown
    - Comfy dtype: `LATENT`
    - Python dtype: `unknown`
- **`vae`**
    - unknown
    - Comfy dtype: `VAE`
    - Python dtype: `unknown`
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
- **`steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`cfg`**
    - Sets the conditioning-free guidance (CFG) scale, adjusting the influence of conditional inputs on the generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Selects the specific sampling algorithm to be used, allowing for customization of the sampling behavior.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Chooses the scheduling algorithm for controlling the sampling process, impacting the progression of noise reduction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Determines the level of denoising applied to the final output, affecting the clarity and quality of the generated image.
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
- **`clip`**
    - unknown
    - Comfy dtype: `CLIP`
    - Python dtype: `unknown`
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
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
- **`model`**
    - Comfy dtype: `MODEL`
    - The generative model used for sampling, central to the generation process.
    - Python dtype: `object`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The positive conditioning applied to guide the generation towards desired attributes or content.
    - Python dtype: `object`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The negative conditioning applied to steer the generation away from undesired attributes or content.
    - Python dtype: `object`
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent representation used or generated during the sampling process.
    - Python dtype: `object`
- **`vae`**
    - Comfy dtype: `VAE`
    - The optional variational autoencoder (VAE) model used in the generation process, enhancing the diversity of outputs.
    - Python dtype: `object`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The optional CLIP model included to guide the generation process based on textual descriptions.
    - Python dtype: `object`
- **`images`**
    - Comfy dtype: `IMAGE`
    - The generated or refined images based on the provided inputs and settings, showcasing the capabilities of the sampling process.
    - Python dtype: `object`
- **`seed`**
    - Comfy dtype: `INT`
    - The seed used for noise generation, ensuring reproducibility of the randomness.
    - Python dtype: `int`
- **`plot_image`**
    - Comfy dtype: `IMAGE`
    - Visual representation of the generation process or outcomes, if applicable.
    - Python dtype: `object`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_KSampler_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),

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
                "optional": {
                    "clip": ("CLIP",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "input_image_override": ("IMAGE",),
                    "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "ttNnodeVersion": ttN_KSampler_v2.version
                },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "latent","vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/base"

    def sample(self, model, positive, negative, latent, vae,
                lora_name, lora_strength,
                steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, denoise=1.0, 
                input_image_override=None,
                clip=None, seed=None, adv_xyPlot=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
                prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        def process_sample_state(model, images, clip, samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                if clip == None:
                    raise ValueError(f"tinyKSampler [{my_unique_id}] - Lora requires CLIP model")
                model, clip = loader.load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            if upscale_method != "None":
                samples = sampler.handle_upscale(samples, upscale_method, factor, crop, upscale_model_name, vae, images, rescale, percent, width, height, longer_side)

            samples = sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      
            results = list()
            if (image_output != "Disabled"):
                # Save images
                latent = samples["samples"]
                images = vae.decode(latent)

                results = ttN_save.images(images, save_prefix, image_output, embed_workflow, file_type)
            
            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (model, positive, negative, samples, vae, clip, images, seed, None)

            return {"ui": {"images": results},
                    "result": (model, positive, negative, samples, vae, clip, images, seed, None)}

        def process_xyPlot(model, clip, samp_samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)
            
            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(model, images, clip, samp_samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            if image_output in ("Hide", "Hide/Save"):
                return (model, positive, negative, samples, vae, clip, images, seed, plot_image)

            return {"ui": {"images": plot_result}, "result": (model, positive, negative, samples, vae, clip, images, seed, plot_image)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(model, input_image_override, clip, latent, vae, seed, positive, negative, lora_name, lora_strength, lora_strength,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent)
        else:
            return process_xyPlot(model, clip, latent, vae, seed, positive, negative, lora_name, lora_strength, lora_strength, steps, cfg, sampler_name, 
                                  scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

```
