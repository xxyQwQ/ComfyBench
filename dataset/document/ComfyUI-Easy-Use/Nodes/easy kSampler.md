---
tags:
- SamplerScheduler
- Sampling
---

# EasyKSampler
## Documentation
- Class name: `easy kSampler`
- Category: `EasyUse/Sampler`
- Output node: `True`

The `easy kSampler` node simplifies the sampling process in generative models by providing a streamlined interface for generating images. It abstracts complex sampling configurations and parameters, offering users an easier way to produce high-quality image outputs.
## Input types
### Required
- **`pipe`**
    - Specifies the pipeline configuration for the sampling process, determining how input data is processed and how outputs are generated.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `str`
- **`image_output`**
    - Controls the output behavior of the sampling process, including whether to preview, save, or send the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`link_id`**
    - A unique identifier for the link in the pipeline, used to manage and track the flow of data.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`save_prefix`**
    - Defines a prefix for saved image files, allowing for organized storage and easy retrieval.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`model`**
    - Specifies the generative model used for sampling. This parameter is optional, allowing for flexibility in model selection.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Represents the updated pipeline configuration after the sampling process, reflecting any changes or adjustments made.
    - Python dtype: `str`
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated image as a result of the sampling process, which can be previewed, saved, or sent based on the specified output behavior.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class samplerSimple(samplerFull):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),
                 "image_output": (["Hide", "Preview", "Preview&Choose", "Save", "Hide&Save", "Sender", "Sender&Save"],{"default": "Preview"}),
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
    FUNCTION = "simple"
    CATEGORY = "EasyUse/Sampler"

    def simple(self, pipe, image_output, link_id, save_prefix, model=None, tile_size=None, prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False):

        return super().run(pipe, None, None, None, None, None, image_output, link_id, save_prefix,
                                 None, model, None, None, None, None, None, None,
                                 None, prompt, extra_pnginfo, my_unique_id, force_full_denoise, disable_noise)

```
