---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# SEGSDetailer
## Documentation
- Class name: `SEGSDetailer`
- Category: `ImpactPack/Detailer`
- Output node: `False`

The SEGSDetailer node is designed to enhance and refine segmentation details within images, focusing on improving the quality and precision of segmentation results. It employs advanced processing techniques to adjust and optimize segmentation data, ensuring higher fidelity and utility for subsequent analysis or applications.
## Input types
### Required
- **`image`**
    - The input image to be detailed or refined. It serves as the foundation for the segmentation detailing process, directly influencing the enhancements applied to the segmentation results.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`segs`**
    - The segmentation data to be enhanced or refined. This input is crucial for the node's operation as it represents the target of the detailing process, affecting the quality and precision of the output.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
- **`guide_size`**
    - Specifies the target size for guiding the detailing process. It plays a significant role in determining the scale and detail level of the output segmentation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Determines the context or specific elements within the image for which the guide size is applicable, influencing the focus and precision of the detailing process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`max_size`**
    - Defines the maximum size limit for the detailing process. It ensures that the output segmentation does not exceed this size, affecting the balance between detail and computational efficiency.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for random number generation, ensuring reproducibility of the detailing process. It influences the variability and consistency of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to execute in the detailing process, affecting the depth of refinement and the quality of the output segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the detailing process, guiding the algorithm's behavior and affecting the outcome's quality.
    - Comfy dtype: `FLOAT`
    - Python dtype: `str`
- **`sampler_name`**
    - Specifies the sampling method to be used in the detailing process, influencing the texture and quality of the detailed segments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler algorithm used to manage the detailing process, affecting the efficiency and outcome of the segmentation refinement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - Indicates whether denoising techniques are applied during the detailing process, improving the clarity and quality of the output segmentation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`noise_mask`**
    - A mask applied to specify areas for noise reduction, focusing the denoising effect and enhancing the detailing precision.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `torch.Tensor`
- **`force_inpaint`**
    - Forces the use of inpainting in areas lacking detail, improving the completeness and quality of the segmentation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`basic_pipe`**
    - The basic processing pipeline configurations used for the detailing process, setting the foundation for the enhancement techniques applied.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `str`
- **`refiner_ratio`**
    - The ratio of refinement applied to the segmentation, adjusting the intensity of detailing and affecting the final output's precision.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`batch_size`**
    - The size of the batch processed in a single detailing operation, affecting the throughput and efficiency of the node.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cycle`**
    - The number of cycles the detailing process is repeated, influencing the depth of refinement and the quality of the output segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`refiner_basic_pipe_opt`**
    - Optional configurations for the refinement pipeline, allowing for customization of the detailing process to achieve specific outcomes.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `str`
- **`inpaint_model`**
    - Indicates whether an inpainting model is used during the detailing process, enabling the filling of missing or incomplete segments for a more complete output.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_mask_feather`**
    - The feathering applied to the noise mask, smoothing the transition between treated and untreated areas, and enhancing the visual quality of the detailed segmentation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scheduler_func_opt`**
    - Optional configurations for the scheduler function, allowing for further customization of the detailing process to optimize performance and results.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `str`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - Outputs the enhanced or refined segmentation data, reflecting the improvements made by the node.
    - Python dtype: `List[SEG]`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - Provides a list of images processed through the control net, useful for visualizing the effects of the detailing process on the segmentation.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SEGSDetailer:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "segs": ("SEGS", ),
                     "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     "max_size": ("FLOAT", {"default": 768, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (core.SCHEDULERS,),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "basic_pipe": ("BASIC_PIPE",),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                     "inpaint_model": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                     "scheduler_func_opt": ("SCHEDULER_FUNC",),
                     }
                }

    RETURN_TYPES = ("SEGS", "IMAGE")
    RETURN_NAMES = ("segs", "cnet_images")
    OUTPUT_IS_LIST = (False, True)

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
                  refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image.shape)

        new_segs = []
        cnet_pil_list = []

        for i in range(batch_size):
            seed += 1
            for seg in segs[1]:
                cropped_image = seg.cropped_image if seg.cropped_image is not None \
                                                  else crop_ndarray4(image.numpy(), seg.crop_region)
                cropped_image = to_tensor(cropped_image)

                is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
                if is_mask_all_zeros:
                    print(f"Detailer: segment skip [empty mask]")
                    new_segs.append(seg)
                    continue

                if noise_mask:
                    cropped_mask = seg.cropped_mask
                else:
                    cropped_mask = None

                cropped_positive = [
                    [condition, {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    }]
                    for condition, details in positive
                ]

                cropped_negative = [
                    [condition, {
                        k: core.crop_condition_mask(v, image, seg.crop_region) if k == "mask" else v
                        for k, v in details.items()
                    }]
                    for condition, details in negative
                ]

                enhanced_image, cnet_pils = core.enhance_detail(cropped_image, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                cropped_positive, cropped_negative, denoise, cropped_mask, force_inpaint,
                                                                refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                                                control_net_wrapper=seg.control_net_wrapper, cycle=cycle,
                                                                inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func=scheduler_func_opt)

                if cnet_pils is not None:
                    cnet_pil_list.extend(cnet_pils)

                if enhanced_image is None:
                    new_cropped_image = cropped_image
                else:
                    new_cropped_image = enhanced_image

                new_seg = SEG(to_numpy(new_cropped_image), seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
                new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_pil_list

    def doit(self, image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio=None, batch_size=1, cycle=1,
             refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        if len(image) > 1:
            raise Exception('[Impact Pack] ERROR: SEGSDetailer does not allow image batches.\nPlease refer to https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/batching-detailer.md for more information.')

        segs, cnet_pil_list = SEGSDetailer.do_detail(image, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                     scheduler, denoise, noise_mask, force_inpaint, basic_pipe, refiner_ratio, batch_size, cycle=cycle,
                                                     refiner_basic_pipe_opt=refiner_basic_pipe_opt,
                                                     inpaint_model=inpaint_model, noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        # set fallback image
        if len(cnet_pil_list) == 0:
            cnet_pil_list = [empty_pil_tensor()]

        return segs, cnet_pil_list

```
