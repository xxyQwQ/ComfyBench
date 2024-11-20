---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# PreDetailerFix
## Documentation
- Class name: `easy preDetailerFix`
- Category: `EasyUse/Fix`
- Output node: `False`

The `easy preDetailerFix` node is designed to prepare images for detailed refinement, adjusting various aspects of the image before it undergoes a more intensive detail enhancement process. This preparation step is crucial for achieving optimal results in subsequent detail enhancement stages, ensuring that images are properly conditioned for the application of advanced detailing techniques.
## Input types
### Required
- **`pipe`**
    - Represents the current state of the image processing pipeline, including models, settings, and intermediate results. It is essential for maintaining continuity and context throughout the detailing process.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict`
- **`guide_size`**
    - Specifies the size of the guide image used in the detailing process, affecting the level of detail that can be achieved.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Indicates whether the guide size is for bounding box or crop region, providing flexibility in the detailing approach.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_size`**
    - Sets the maximum size limit for the images being processed, ensuring that the detailing operations are performed within resource constraints.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`seed`**
    - Provides a seed value for any random operations within the detailing process, ensuring reproducibility of results.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the number of steps to be taken in the detailing process, directly impacting the intensity and duration of the detailing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configures the detailing process parameters, allowing for customization of the enhancement techniques applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampling method to be used in the detailing process, affecting the quality and characteristics of the detail enhancement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Selects the scheduling algorithm for the detailing process, optimizing the sequence and timing of operations for efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Enables or disables denoising in the detailing process, affecting the clarity and quality of the final image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`feather`**
    - Adjusts the feathering of edges in the detailing process, smoothing transitions for a more natural appearance.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_mask`**
    - Specifies whether a noise mask is applied, adding texture and realism to the enhanced details.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`force_inpaint`**
    - Forces the use of inpainting in areas where detail enhancement is difficult, ensuring consistent quality across the image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`drop_size`**
    - Determines the size of drops to be used in the detailing process, affecting the texture and detail of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`wildcard`**
    - Allows for the inclusion of wildcard patterns for advanced detailing customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`cycle`**
    - Specifies the number of cycles the detailing process will run, affecting the depth of detail enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`bbox_segm_pipe`**
    - Provides the bounding box and segmentation pipeline settings for detail enhancement.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict`
- **`sam_pipe`**
    - Specifies the settings for the SAM pipeline used in detail enhancement.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `Dict`
- **`optional_image`**
    - Allows for the inclusion of an optional image to be used in the detailing process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - Outputs the updated image processing pipeline, including any modifications or enhancements made during the detailing process.
    - Python dtype: `Dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class preDetailerFix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
             "pipe": ("PIPE_LINE",),
             "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
             "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
             "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
             "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
             "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
             "scheduler": (comfy.samplers.KSampler.SCHEDULERS + ['align_your_steps'],),
             "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
             "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
             "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
             "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
             "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),
             "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
        },
            "optional": {
                "bbox_segm_pipe": ("PIPE_LINE",),
                "sam_pipe": ("PIPE_LINE",),
                "optional_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("PIPE_LINE",)
    RETURN_NAMES = ("pipe",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "doit"

    CATEGORY = "EasyUse/Fix"

    def doit(self, pipe, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler, denoise, feather, noise_mask, force_inpaint, drop_size, wildcard, cycle, bbox_segm_pipe=None, sam_pipe=None, optional_image=None):

        model = pipe["model"] if "model" in pipe else None
        if model is None:
            raise Exception(f"[ERROR] pipe['model'] is missing")
        clip = pipe["clip"] if"clip" in pipe else None
        if clip is None:
            raise Exception(f"[ERROR] pipe['clip'] is missing")
        vae = pipe["vae"] if "vae" in pipe else None
        if vae is None:
            raise Exception(f"[ERROR] pipe['vae'] is missing")
        if optional_image is not None:
            images = optional_image
        else:
            images = pipe["images"] if "images" in pipe else None
            if images is None:
                raise Exception(f"[ERROR] pipe['image'] is missing")
        positive = pipe["positive"] if "positive" in pipe else None
        if positive is None:
            raise Exception(f"[ERROR] pipe['positive'] is missing")
        negative = pipe["negative"] if "negative" in pipe else None
        if negative is None:
            raise Exception(f"[ERROR] pipe['negative'] is missing")
        bbox_segm_pipe = bbox_segm_pipe or (pipe["bbox_segm_pipe"] if pipe and "bbox_segm_pipe" in pipe else None)
        if bbox_segm_pipe is None:
            raise Exception(f"[ERROR] bbox_segm_pipe or pipe['bbox_segm_pipe'] is missing")
        sam_pipe = sam_pipe or (pipe["sam_pipe"] if pipe and "sam_pipe" in pipe else None)
        if sam_pipe is None:
            raise Exception(f"[ERROR] sam_pipe or pipe['sam_pipe'] is missing")

        loader_settings = pipe["loader_settings"] if "loader_settings" in pipe else {}

        if(scheduler == 'align_your_steps'):
            model_version = get_sd_version(model)
            if model_version == 'sdxl':
                scheduler = 'AYS SDXL'
            elif model_version == 'svd':
                scheduler = 'AYS SVD'
            else:
                scheduler = 'AYS SD1'

        new_pipe = {
            "images": images,
            "model": model,
            "clip": clip,
            "vae": vae,
            "positive": positive,
            "negative": negative,
            "seed": seed,

            "bbox_segm_pipe": bbox_segm_pipe,
            "sam_pipe": sam_pipe,

            "loader_settings": loader_settings,

            "detail_fix_settings": {
                "guide_size": guide_size,
                "guide_size_for": guide_size_for,
                "max_size": max_size,
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "denoise": denoise,
                "feather": feather,
                "noise_mask": noise_mask,
                "force_inpaint": force_inpaint,
                "drop_size": drop_size,
                "wildcard": wildcard,
                "cycle": cycle
            }
        }


        del bbox_segm_pipe
        del sam_pipe

        return (new_pipe,)

```
