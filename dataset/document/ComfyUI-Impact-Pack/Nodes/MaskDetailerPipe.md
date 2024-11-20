---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# MaskDetailer (pipe)
## Documentation
- Class name: `MaskDetailerPipe`
- Category: `ImpactPack/Detailer`
- Output node: `False`

The MaskDetailerPipe node is designed to enhance and refine the details of masks within images, leveraging advanced image processing techniques to improve the visual quality and accuracy of the masks. It operates within a pipeline to apply detailed adjustments and refinements to mask elements, aiming to achieve a higher level of clarity and definition in the output images.
## Input types
### Required
- **`image`**
    - The primary image input for mask detailing, serving as the basis for enhancement and refinement operations.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask`**
    - The mask input that specifies the areas within the image to be detailed and refined, playing a crucial role in the detailing process.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`basic_pipe`**
    - A tuple containing the models and configurations used for the initial processing and enhancement of the image and mask.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Tuple`
- **`guide_size`**
    - Specifies the target size for guiding the detail enhancement process, affecting the scale and precision of the detailing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Determines the specific application or context for the guide size, influencing how the detailing is optimized.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_size`**
    - The maximum size limit for the image processing, ensuring that the detailing does not exceed computational constraints.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`mask_mode`**
    - Indicates whether the detailing should focus only on masked areas or include the whole image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`seed`**
    - The seed value for random number generation, ensuring reproducibility of the detailing process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to execute in the detailing process, defining the depth of detail enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the detailing process, influencing the behavior and outcome of the enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler used in the detailing process, affecting the pattern and quality of detail added.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler used to manage the progression of steps in the detailing process, optimizing the enhancement workflow.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - The level of denoising applied during the detailing process, affecting the smoothness of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`feather`**
    - The feathering amount applied to the edges of the mask, softening transitions for a more natural look.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_factor`**
    - The factor by which the image is cropped around the mask, affecting the focus area of the detailing.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`drop_size`**
    - The minimum size of details to be dropped, controlling the level of detail refinement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`refiner_ratio`**
    - The ratio of refinement applied to the detailing process, adjusting the intensity of enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`batch_size`**
    - The number of images processed in a batch, relevant for batch processing scenarios.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cycle`**
    - The number of cycles the detailing process is repeated, affecting the depth of enhancement.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`refiner_basic_pipe_opt`**
    - An optional tuple for refinement settings, similar to basic_pipe but for the refinement stage.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Tuple`
- **`detailer_hook`**
    - A hook for custom detailing operations, allowing for additional processing steps.
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `Callable`
- **`inpaint_model`**
    - Indicates whether an inpainting model is used for filling masked areas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_mask_feather`**
    - The feathering applied to the noise mask, affecting the blending of noise in detailed areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`bbox_fill`**
    - Determines if bounding boxes are filled, affecting the handling of segmented areas.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`contour_fill`**
    - Indicates whether contours are filled, affecting the detailing of edges and boundaries.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`scheduler_func_opt`**
    - An optional scheduler function for advanced control over the detailing process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `Callable`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced image with refined mask details, showcasing improved visual quality and accuracy.
    - Python dtype: `torch.Tensor`
- **`cropped_refined`**
    - Comfy dtype: `IMAGE`
    - A cropped and further refined version of the mask, focusing on specific areas for detailed enhancement.
    - Python dtype: `torch.Tensor`
- **`cropped_enhanced_alpha`**
    - Comfy dtype: `IMAGE`
    - The alpha channel of the cropped and enhanced image, providing transparency information for further processing.
    - Python dtype: `torch.Tensor`
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The tuple of models and configurations returned after processing, potentially updated from the input state.
    - Python dtype: `Tuple`
- **`refiner_basic_pipe_opt`**
    - Comfy dtype: `BASIC_PIPE`
    - An optional tuple similar to basic_pipe, but specifically for the refinement process, indicating any changes or additional configurations used.
    - Python dtype: `Tuple`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskDetailerPipe:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "image": ("IMAGE", ),
                    "mask": ("MASK", ),
                    "basic_pipe": ("BASIC_PIPE",),

                    "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "mask bbox", "label_off": "crop region"}),
                    "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                    "mask_mode": ("BOOLEAN", {"default": True, "label_on": "masked only", "label_off": "whole"}),

                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (core.SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),

                    "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                    "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),
                    "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),
                    "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                    "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),

                    "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                   },
                "optional": {
                    "refiner_basic_pipe_opt": ("BASIC_PIPE", ),
                    "detailer_hook": ("DETAILER_HOOK",),
                    "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                    "bbox_fill": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                    "contour_fill": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                    "scheduler_func_opt": ("SCHEDULER_FUNC",),
                   }
                }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "BASIC_PIPE", "BASIC_PIPE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "basic_pipe", "refiner_basic_pipe_opt")
    OUTPUT_IS_LIST = (False, True, True, False, False)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    def doit(self, image, mask, basic_pipe, guide_size, guide_size_for, max_size, mask_mode,
             seed, steps, cfg, sampler_name, scheduler, denoise,
             feather, crop_factor, drop_size, refiner_ratio, batch_size, cycle=1,
             refiner_basic_pipe_opt=None, detailer_hook=None, inpaint_model=False, noise_mask_feather=0,
             bbox_fill=False, contour_fill=True, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: MaskDetailer does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        model, clip, vae, positive, negative = basic_pipe

        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        # create segs
        if mask is not None:
            mask = make_2d_mask(mask)
            segs = core.mask_to_segs(mask, False, crop_factor, bbox_fill, drop_size, is_contour=contour_fill)
        else:
            segs = ((image.shape[1], image.shape[2]), [])

        enhanced_img_batch = None
        cropped_enhanced_list = []
        cropped_enhanced_alpha_list = []

        for i in range(batch_size):
            if mask is not None:
                enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, _, _ = \
                    DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for, max_size, seed+i, steps,
                                              cfg, sampler_name, scheduler, positive, negative, denoise, feather, mask_mode,
                                              force_inpaint=True, wildcard_opt=None, detailer_hook=detailer_hook,
                                              refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip,
                                              refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                              cycle=cycle, inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)
            else:
                enhanced_img, cropped_enhanced, cropped_enhanced_alpha = image, [], []

            if enhanced_img_batch is None:
                enhanced_img_batch = enhanced_img
            else:
                enhanced_img_batch = torch.cat((enhanced_img_batch, enhanced_img), dim=0)

            cropped_enhanced_list += cropped_enhanced
            cropped_enhanced_alpha_list += cropped_enhanced_alpha

        # set fallback image
        if len(cropped_enhanced_list) == 0:
            cropped_enhanced_list = [empty_pil_tensor()]

        if len(cropped_enhanced_alpha_list) == 0:
            cropped_enhanced_alpha_list = [empty_pil_tensor()]

        return enhanced_img_batch, cropped_enhanced_list, cropped_enhanced_alpha_list, basic_pipe, refiner_basic_pipe_opt

```
