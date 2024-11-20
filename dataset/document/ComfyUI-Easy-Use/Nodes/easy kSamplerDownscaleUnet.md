---
tags:
- ImageScaling
---

# EasyKsampler (Downscale Unet)
## Documentation
- Class name: `easy kSamplerDownscaleUnet`
- Category: `EasyUse/Sampler`
- Output node: `True`

This node specializes in simplifying the process of image sampling by integrating Unet downscaling capabilities. It aims to enhance efficiency and performance in image generation tasks by applying a downscaling factor to the Unet model, which is particularly useful for handling high-resolution images or complex sampling scenarios. The node facilitates a streamlined workflow for image sampling with an emphasis on reducing computational load without compromising the quality of the generated images.
## Input types
### Required
- **`pipe`**
    - The pipeline through which the image data flows, serving as the primary conduit for processing and transforming images within the node.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict[str, Any]`
- **`downscale_mode`**
    - Determines the mode of downscaling to be applied to the Unet model, affecting how the model's resolution is reduced.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`block_number`**
    - Specifies the block number within the Unet model to which the downscaling is applied, targeting specific layers for resolution reduction.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`downscale_factor`**
    - The factor by which the Unet model's resolution is reduced, directly influencing the detail and size of the output images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - Defines the starting percentage of the model's depth at which downscaling begins, allowing for selective downscaling depth.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Sets the ending percentage of the model's depth where downscaling concludes, further refining the downscaling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`downscale_after_skip`**
    - A boolean parameter that determines whether downscaling occurs after skip connections in the Unet model, affecting the model's internal processing flow.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`downscale_method`**
    - Specifies the method used for reducing the resolution of the Unet model, impacting the quality and characteristics of the downscaled output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`upscale_method`**
    - The method used to upscale the model back to its original or a specified resolution, complementing the downscaling process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`image_output`**
    - Controls the output format of the images, including options for previewing, saving, or hiding the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`link_id`**
    - An identifier used for linking the node's output with other processes or outputs, facilitating integration within larger workflows.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_prefix`**
    - A prefix applied to filenames when saving output images, aiding in the organization and retrieval of generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`model`**
    - The model to be downscaled, typically a Unet model, which is central to the node's operation and directly affects the output quality.
    - Comfy dtype: `MODEL`
    - Python dtype: `Optional[torch.nn.Module]`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified pipeline after processing, which includes the downscaled Unet model and potentially other transformations applied to the image data.
    - Python dtype: `Dict[str, Any]`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final image output after downscaling, reflecting the applied adjustments and ready for further use or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerSimpleDownscaleUnet(samplerFull):

    upscale_methods = ["bicubic", "nearest-exact", "bilinear", "area", "bislerp"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "downscale_mode": (["None", "Auto", "Custom"],{"default": "Auto"}),
                 "block_number": ("INT", {"default": 3, "min": 1, "max": 32, "step": 1}),
                 "downscale_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 9.0, "step": 0.001}),
                 "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "end_percent": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.001}),
                 "downscale_after_skip": ("BOOLEAN", {"default": True}),
                 "downscale_method": (s.upscale_methods,),
                 "upscale_method": (s.upscale_methods,),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "downscale_unet"
    CATEGORY = "EasyUse/Sampler"

    def downscale_unet(self, pipe, downscale_mode, block_number, downscale_factor, start_percent, end_percent, downscale_after_skip, downscale_method, upscale_method, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):
        downscale_options = None
        if downscale_mode == 'Auto':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": None,
                "start_percent": 0,
                "end_percent":0.35,
                "downscale_after_skip": True,
                "downscale_method": "bicubic",
                "upscale_method": "bicubic"
            }
        elif downscale_mode == 'Custom':
            downscale_options = {
                "block_number": block_number,
                "downscale_factor": downscale_factor,
                "start_percent": start_percent,
                "end_percent": end_percent,
                "downscale_after_skip": downscale_after_skip,
                "downscale_method": downscale_method,
                "upscale_method": upscale_method
            }

        return super().run(pipe, None, None,None,None,None, image_output, link_id, save_prefix,
                               None, model, None, None, None, None, None, None,
                               tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise, downscale_options)

```
