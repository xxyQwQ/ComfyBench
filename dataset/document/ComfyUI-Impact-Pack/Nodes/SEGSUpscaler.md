---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# Upscaler (SEGS)
## Documentation
- Class name: `SEGSUpscaler`
- Category: `ImpactPack/Upscale`
- Output node: `False`

The SEGSUpscaler node is designed to upscale images by segmenting them into smaller sections, enhancing each segment individually using a combination of models and techniques, and then reassembling the enhanced segments back into a single, upscaled image. This process allows for more detailed and controlled upscaling, leveraging the power of segmentation to address specific areas of an image for improvement.
## Input types
### Required
- **`image`**
    - The original image to be upscaled. It serves as the base for segmentation and subsequent upscaling processes.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`segs`**
    - A collection of image segments, each representing a portion of the original image to be individually upscaled and enhanced.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
- **`model`**
    - The primary model used for processing and enhancing the image segments.
    - Comfy dtype: `MODEL`
    - Python dtype: `Model`
- **`clip`**
    - The CLIP model used for guiding the enhancement process based on textual descriptions.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
- **`vae`**
    - The VAE model used for encoding and decoding image segments during the upscaling process.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`rescale_factor`**
    - The factor by which the image will be rescaled as part of the upscaling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resampling_method`**
    - The method used for resampling the image during the upscaling process, affecting the quality and characteristics of the upscaled image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`supersample`**
    - Indicates whether supersampling is applied to the image for enhanced upscaling quality.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`rounding_modulus`**
    - A value used to adjust the dimensions of the upscaled image, ensuring they are multiples of a specific number.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the upscaling process across segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the image enhancement process for each segment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the image enhancement process, guiding the behavior of the models used.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampling method used in the image enhancement process, affecting the generation of enhanced segments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler used to control the progression of steps in the image enhancement process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `Scheduler`
- **`positive`**
    - Positive text prompts used to guide the image enhancement process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative text prompts used to counterbalance the positive prompts, refining the enhancement process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`denoise`**
    - A parameter controlling the level of denoising applied to each image segment during enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`feather`**
    - The feathering value applied to the edges of image segments, smoothing transitions between enhanced segments and the rest of the image.
    - Comfy dtype: `INT`
    - Python dtype: `float`
- **`inpaint_model`**
    - The model used for inpainting, filling in areas of the image that may be missing or require correction after segmentation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `InpaintModel`
- **`noise_mask_feather`**
    - The feathering value applied to the noise mask, smoothing the application of noise adjustments across image segments.
    - Comfy dtype: `INT`
    - Python dtype: `float`
### Optional
- **`upscale_model_opt`**
    - An optional model specifically designed for upscaling, providing an alternative or supplementary method to the main upscaling process.
    - Comfy dtype: `UPSCALE_MODEL`
    - Python dtype: `Optional[UpscaleModel]`
- **`upscaler_hook_opt`**
    - An optional hook allowing for custom post-processing steps after each segment is upscaled and reassembled into the final image.
    - Comfy dtype: `UPSCALER_HOOK`
    - Python dtype: `Optional[PostProcessHook]`
- **`scheduler_func_opt`**
    - An optional scheduler function providing additional control over the scheduling of enhancement steps for each segment.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `Optional[SchedulerFunction]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The final, upscaled image, composed of individually enhanced and reassembled image segments.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SEGSUpscaler:
    @classmethod
    def INPUT_TYPES(s):
        resampling_methods = ["lanczos", "nearest", "bilinear", "bicubic"]

        return {"required": {
                    "image": ("IMAGE",),
                    "segs": ("SEGS",),
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "vae": ("VAE",),
                    "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 100.0, "step": 0.01}),
                    "resampling_method": (resampling_methods,),
                    "supersample": (["true", "false"],),
                    "rounding_modulus": ("INT", {"default": 8, "min": 8, "max": 1024, "step": 8}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (core.SCHEDULERS,),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    },
                "optional": {
                    "upscale_model_opt": ("UPSCALE_MODEL",),
                    "upscaler_hook_opt": ("UPSCALER_HOOK",),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                    }
                }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Upscale"

    @staticmethod
    def doit(image, segs, model, clip, vae, rescale_factor, resampling_method, supersample, rounding_modulus,
             seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, feather, inpaint_model, noise_mask_feather,
             upscale_model_opt=None, upscaler_hook_opt=None, scheduler_func_opt=None):

        new_image = segs_upscaler.upscaler(image, upscale_model_opt, rescale_factor, resampling_method, supersample, rounding_modulus)

        segs = core.segs_scale_match(segs, new_image.shape)

        ordered_segs = segs[1]

        for i, seg in enumerate(ordered_segs):
            cropped_image = crop_ndarray4(new_image.numpy(), seg.crop_region)
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                print(f"SEGSUpscaler: segment skip [empty mask]")
                continue

            cropped_mask = seg.cropped_mask

            seg_seed = seed + i

            enhanced_image = segs_upscaler.img2img_segs(cropped_image, model, clip, vae, seg_seed, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, denoise,
                                                        noise_mask=cropped_mask, control_net_wrapper=seg.control_net_wrapper,
                                                        inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)
            if not (enhanced_image is None):
                new_image = new_image.cpu()
                enhanced_image = enhanced_image.cpu()
                left = seg.crop_region[0]
                top = seg.crop_region[1]
                tensor_paste(new_image, enhanced_image, (left, top), mask)

                if upscaler_hook_opt is not None:
                    new_image = upscaler_hook_opt.post_paste(new_image)

        enhanced_img = tensor_convert_rgb(new_image)

        return (enhanced_img,)

```
