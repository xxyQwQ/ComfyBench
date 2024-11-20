---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# SEGSDetailer For AnimateDiff (SEGS/pipe)
## Documentation
- Class name: `SEGSDetailerForAnimateDiff`
- Category: `ImpactPack/Detailer`
- Output node: `False`

This node specializes in enhancing the detail and quality of segmentation maps specifically for animated differences, utilizing a detailed processing pipeline to refine and adjust segmentation outputs for improved animation effects. It operates by taking in image frames and segmentation maps, applying a series of refinements, and outputting enhanced segmentation maps along with corresponding control net images.
## Input types
### Required
- **`image_frames`**
    - The sequence of image frames to be processed. It plays a crucial role in guiding the detail enhancement process by providing the visual context necessary for refining the segmentation maps.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`segs`**
    - The initial segmentation maps that are to be enhanced. These maps are crucial for the node's operation as they serve as the base upon which refinements are applied to improve the animation effect.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[torch.Tensor]`
- **`guide_size`**
    - Specifies the target size for guiding the detail enhancement process. It affects the scaling of the input segmentation maps and image frames, impacting the precision and quality of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`guide_size_for`**
    - Indicates the specific purpose or target for the guide size, affecting how the detail enhancement process is tailored to different animation requirements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `str`
- **`max_size`**
    - The maximum allowable size for the processed outputs, ensuring that the enhanced segmentation maps and control net images do not exceed this limit, which helps in managing resource utilization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`seed`**
    - A seed value for ensuring reproducibility in the detail enhancement process. It influences the randomness in the process, allowing for consistent results across runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to execute in the detail enhancement process, affecting the depth of refinement applied to the segmentation maps and the overall quality of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the detail enhancement process, providing flexibility in adjusting the parameters and algorithms used for refining the segmentation maps.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Dict[str, Any]`
- **`sampler_name`**
    - Specifies the sampling method to be used in the detail enhancement process, influencing the approach to refining the segmentation maps and control net images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler to be used in the detail enhancement process, coordinating the execution of steps and the application of refinements to the segmentation maps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - A flag indicating whether denoising should be applied during the detail enhancement process, which can help in improving the clarity and quality of the output segmentation maps.
    - Comfy dtype: `FLOAT`
    - Python dtype: `bool`
- **`basic_pipe`**
    - Specifies the basic processing pipeline to be used for the initial stages of the detail enhancement process, setting the foundation for further refinements.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `str`
- **`refiner_ratio`**
    - The ratio of refinement to be applied, determining the extent of detail enhancement in relation to the initial processing. It allows for fine-tuning the balance between refinement and original detail.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`refiner_basic_pipe_opt`**
    - Optional configurations for the basic processing pipeline during the refinement stage, offering additional customization for the detail enhancement process.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `Dict[str, Any]`
- **`noise_mask_feather`**
    - The feathering value for noise masks, affecting the smoothness of transitions and blending in the enhanced segmentation maps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scheduler_func_opt`**
    - Optional configurations for the scheduler function, providing further customization for the coordination and execution of the detail enhancement process.
    - Comfy dtype: `SCHEDULER_FUNC`
    - Python dtype: `Dict[str, Any]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The enhanced segmentation maps, which have been refined and adjusted for improved animation effects, showcasing the effectiveness of the detail enhancement process.
    - Python dtype: `List[torch.Tensor]`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - The control net images corresponding to the enhanced segmentation maps, which are used to guide further processing and refinement for animation purposes.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [SEGSPaste](../../ComfyUI-Impact-Pack/Nodes/SEGSPaste.md)



## Source code
```python
class SEGSDetailerForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                     "image_frames": ("IMAGE", ),
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
                     "basic_pipe": ("BASIC_PIPE",),
                     "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                     },
                "optional": {
                     "refiner_basic_pipe_opt": ("BASIC_PIPE",),
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
    def do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                  denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, noise_mask_feather=0, scheduler_func_opt=None):

        model, clip, vae, positive, negative = basic_pipe
        if refiner_basic_pipe_opt is None:
            refiner_model, refiner_clip, refiner_positive, refiner_negative = None, None, None, None
        else:
            refiner_model, refiner_clip, _, refiner_positive, refiner_negative = refiner_basic_pipe_opt

        segs = core.segs_scale_match(segs, image_frames.shape)

        new_segs = []
        cnet_image_list = []

        for seg in segs[1]:
            cropped_image_frames = None

            for image in image_frames:
                image = image.unsqueeze(0)
                cropped_image = seg.cropped_image if seg.cropped_image is not None else crop_tensor4(image, seg.crop_region)
                cropped_image = to_tensor(cropped_image)
                if cropped_image_frames is None:
                    cropped_image_frames = cropped_image
                else:
                    cropped_image_frames = torch.concat((cropped_image_frames, cropped_image), dim=0)

            cropped_image_frames = cropped_image_frames.cpu().numpy()

            # It is assumed that AnimateDiff does not support conditioning masks based on test results, but it will be added for future consideration.
            cropped_positive = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in positive
            ]

            cropped_negative = [
                [condition, {
                    k: core.crop_condition_mask(v, cropped_image_frames, seg.crop_region) if k == "mask" else v
                    for k, v in details.items()
                }]
                for condition, details in negative
            ]

            enhanced_image_tensor, cnet_images = core.enhance_detail_for_animatediff(cropped_image_frames, model, clip, vae, guide_size, guide_size_for, max_size,
                                                                                     seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                                                     cropped_positive, cropped_negative, denoise, seg.cropped_mask,
                                                                                     refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                                                     refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                                                                     refiner_negative=refiner_negative, control_net_wrapper=seg.control_net_wrapper,
                                                                                     noise_mask_feather=noise_mask_feather, scheduler_func=scheduler_func_opt)
            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if enhanced_image_tensor is None:
                new_cropped_image = cropped_image_frames
            else:
                new_cropped_image = enhanced_image_tensor.cpu().numpy()

            new_seg = SEG(new_cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
            new_segs.append(new_seg)

        return (segs[0], new_segs), cnet_image_list

    def doit(self, image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, basic_pipe, refiner_ratio=None, refiner_basic_pipe_opt=None, inpaint_model=False, noise_mask_feather=0, scheduler_func_opt=None):

        segs, cnet_images = SEGSDetailerForAnimateDiff.do_detail(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name,
                                                                 scheduler, denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt,
                                                                 noise_mask_feather=noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

        if len(cnet_images) == 0:
            cnet_images = [empty_pil_tensor()]

        return (segs, cnet_images)

```
