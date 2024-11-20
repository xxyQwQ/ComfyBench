---
tags:
- Sampling
---

# pipeKSamplerAdvanced
## Documentation
- Class name: `ttN pipeKSamplerAdvanced_v2`
- Category: `🌏 tinyterra/pipe`
- Output node: `True`

This node is designed for advanced sampling in image generation pipelines, incorporating various enhancements and options for customization. It leverages a combination of models, embeddings, and sampling techniques to refine and generate high-quality images based on specified parameters, offering extensive control over the image generation process.
## Input types
### Required
- **`pipe`**
    - Represents the current state of the image generation pipeline, including models, embeddings, and other relevant settings, serving as the foundation for the sampling process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`lora_name`**
    - Specifies the name of the LoRA model to be used for adjusting model weights dynamically, enhancing the generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_strength`**
    - Determines the strength of the LoRA adjustment to the model, allowing for fine-tuning of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`upscale_method`**
    - Specifies the method to be used for upscaling the generated images, enhancing their resolution.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_model_name`**
    - Identifies the specific model to be used for upscaling, affecting the quality of the upscaled images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`factor`**
    - Determines the factor by which the images are upscaled, directly influencing the final image size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`rescale`**
    - Controls whether the generated images are rescaled, affecting their dimensions and aspect ratio.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`percent`**
    - Specifies the percentage by which the images are rescaled, offering precise control over the output size.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`width`**
    - Sets the target width for rescaled images, defining their horizontal dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the target height for rescaled images, defining their vertical dimension.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`longer_side`**
    - Specifies the length of the longer side for rescaled images, ensuring a balanced aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop`**
    - Determines whether and how the generated images are cropped, affecting their composition and focus.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`add_noise`**
    - Controls whether noise is added to the generation process, affecting the texture and details of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`noise`**
    - Specifies the amount and type of noise to be added, if any, during the image generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Specifies the number of steps to be taken in the sampling process, impacting the detail and quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_at_step`**
    - Specifies the starting step for the generation process, allowing for mid-process intervention or customization.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Defines the ending step for the generation process, determining the point at which the process concludes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration setting that influences the sampling behavior, offering additional customization of the image generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Identifies the specific sampling algorithm to be used, affecting the style and characteristics of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling algorithm for the sampling process, influencing the progression and variation of image generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`return_with_leftover_noise`**
    - Controls whether leftover noise is included in the final output, affecting the texture and detail of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`image_output`**
    - Specifies the mode of output for the generated images, including options for hiding or saving the images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`save_prefix`**
    - Defines a prefix for saved image files, organizing the output in a structured manner.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`file_type`**
    - Specifies the file type for saved images, allowing for flexibility in the output format.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`embed_workflow`**
    - Indicates whether the embedding workflow is included in the generation process, affecting the input and refinement of images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`noise_seed`**
    - Sets a specific seed for noise generation, ensuring reproducibility of the added noise effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`optional_model`**
    - Allows for the specification of an alternative model to be used in the generation process, offering flexibility in the approach.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`optional_positive`**
    - Specifies optional positive embeddings to be used, enhancing the generation towards desired attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_negative`**
    - Specifies optional negative embeddings to be used, steering the generation away from certain attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`optional_latent`**
    - Allows for the inclusion of specific latent vectors, directly influencing the starting point of the generation process.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`optional_vae`**
    - Specifies an optional VAE model to be used for decoding or refining the generated images.
    - Comfy dtype: `VAE`
    - Python dtype: `str`
- **`optional_clip`**
    - Specifies an optional CLIP model to be used for guiding the generation process towards textual descriptions.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`input_image_override`**
    - Allows for an existing image to be used as the starting point for the generation process, overriding the default behavior.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`adv_xyPlot`**
    - Enables advanced plotting options for visualizing the generation process, offering insights into the sampling dynamics.
    - Comfy dtype: `ADV_XYPLOT`
    - Python dtype: `str`
## Output types
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
- **`image`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`seed`**
    - Comfy dtype: `INT`
    - unknown
    - Python dtype: `unknown`
- **`ui`**
    - Provides a user interface component for visualizing the generated images, offering an interactive way to review and adjust the output.
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeKSamplerAdvanced_v2:
    version = '2.3.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                
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
                
                "add_noise": (["enable", "disable"], ),
                "noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "return_with_leftover_noise": (["disable", "enable"], ),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
            "optional": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_pipeKSamplerAdvanced_v2.version
                },
            }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", )
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "adv_sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def adv_sample(self, pipe,
               lora_name, lora_strength,
               add_noise, steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, noise, 
               noise_seed=None, optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None, adv_xyPlot=None, upscale_method=None, upscale_model_name=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_at_step=None, end_at_step=None, return_with_leftover_noise=False):

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        return ttN_pipeKSampler_v2.sample(self, pipe, lora_name, lora_strength, steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, noise, 
                optional_model, optional_positive, optional_negative, optional_latent, optional_vae, optional_clip, input_image_override, noise_seed, adv_xyPlot, upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop, prompt, extra_pnginfo, my_unique_id, start_at_step, end_at_step, force_full_denoise, disable_noise)

```
