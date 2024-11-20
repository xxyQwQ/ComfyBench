---
tags:
- Sampling
---

# EasyCascadeKsampler
## Documentation
- Class name: `easy cascadeKSampler`
- Category: `EasyUse/Sampler`
- Output node: `True`

The easy cascadeKSampler node facilitates the generation of images through a cascading sampling process, leveraging a pipeline of models for enhanced detail and quality. It simplifies the complex setup typically required for cascading sampling, making it accessible for users to produce high-quality images with minimal configuration.
## Input types
### Required
- **`pipe`**
    - Specifies the pipeline configuration for the cascading sampling process, including model and processing steps. It's crucial for defining the sequence of operations and models used.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `str`
- **`image_output`**
    - Determines how the output images are handled, offering options for hiding, previewing, saving, or sending the images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`link_id`**
    - An identifier used for linking the output with external systems or files, especially when sending or saving images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_prefix`**
    - A prefix added to saved images, allowing for organized storage and easy retrieval.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`model_c`**
    - An optional model configuration that can be provided to customize the sampling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Returns the updated pipeline configuration after the cascading sampling process, reflecting any changes or additions.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image as a result of the cascading sampling process.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerCascadeSimple(samplerCascadeFull):

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Save", "Hide&Save", "Sender", "Sender&Save"], {"default": "Preview"}),
                 "link_id": ("INT", {"default": 0, "min": 0, "max": sys.maxsize, "step": 1}),
                 "save_prefix": ("STRING", {"default": "ComfyUI"}),
                 },
                "optional": {
                    "model_c": ("MODEL",),
                },
                "hidden":
                  {"tile_size": "INT", "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),)
                  }
                }


    RETURN_TYPES = ("PIPE_LINE", "IMAGE",)
    RETURN_NAMES = ("pipe", "image",)
    OUTPUT_NODE = True
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model_c=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None,None, None,None,None,None, image_output, link_id, save_prefix,
                               None, None, None, model_c, tile_size, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

```
