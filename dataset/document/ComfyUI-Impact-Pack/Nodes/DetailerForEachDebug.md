---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# DetailerDebug (SEGS)
## Documentation
- Class name: `DetailerForEachDebug`
- Category: `ImpactPack/Detailer`
- Output node: `False`

The DetailerForEachDebug node is designed for debugging purposes within the ComfyUI Impact Pack. It extends the functionality of the DetailerForEach node by incorporating additional testing and debugging features to assist developers in identifying and resolving issues related to detail enhancement processes in images or segments.
## Input types
### Required
- **`image`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`segs`**
    - Provides segmentation information for the images, used to guide the detailing process in a more targeted manner.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SegmentationInfo]`
- **`model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`clip`**
    - unknown
    - Comfy dtype: `CLIP`
    - Python dtype: `unknown`
- **`vae`**
    - unknown
    - Comfy dtype: `VAE`
    - Python dtype: `unknown`
- **`guide_size`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`guide_size_for`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`max_size`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`seed`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
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
- **`positive`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`negative`**
    - unknown
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `unknown`
- **`denoise`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`noise_mask`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`force_inpaint`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`wildcard`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`cycle`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
- **`detailer_hook`**
    - unknown
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `unknown`
- **`inpaint_model`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`noise_mask_feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`scheduler_func_opt`**
    - unknown
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `unknown`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Returns the detailed image frames after the debugging process, showcasing the effects of the applied detailing enhancements.
    - Python dtype: `List[Image]`
- **`cropped`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`cropped_refined`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`cropped_refined_alpha`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [DetailerForEachDebug](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebug.md)



## Source code
```python
class DetailerForEachTest(DetailerForEach):
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "cropped", "cropped_refined", "cropped_refined_alpha", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, True, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
             scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, wildcard, detailer_hook=None,
             cycle=1, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: DetailerForEach does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        enhanced_img, cropped, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
            DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed, steps,
                                      cfg, sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                      force_inpaint, wildcard, detailer_hook,
                                      cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        # set fallback image
        if len(cropped) == 0:
            cropped = [empty_pil_tensor()]

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return enhanced_img, cropped, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list

```
