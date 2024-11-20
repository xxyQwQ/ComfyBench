---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# Detailer For AnimateDiff (SEGS/pipe)
## Documentation
- Class name: `DetailerForEachPipeForAnimateDiff`
- Category: `ImpactPack/Detailer`
- Output node: `False`

This node is designed to apply detailed animation effects to each frame or segment within a pipeline, enhancing the visual impact and dynamic elements of the content. It operates by iterating over each applicable segment or frame, applying a set of predefined or dynamically generated animation details to improve the overall aesthetic and engagement of the animation.
## Input types
### Required
- **`image_frames`**
    - The sequence of image frames to which animation effects will be applied. This input is crucial for determining the visual flow and continuity of the animation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[Image]`
- **`segs`**
    - Segmentation information for the image frames, used to identify and apply detailed effects to specific areas of the animation.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[Segmentation]`
- **`guide_size`**
    - Specifies the size guide for the animation detailing process, influencing the scale at which details are applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`guide_size_for`**
    - A boolean flag indicating whether the guide size applies to the bounding box or the crop region, affecting how details are scaled and applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`max_size`**
    - The maximum size limit for the animation details, ensuring that the detailing process stays within computational and aesthetic bounds.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`seed`**
    - A seed value for random number generation, providing consistency in the detailing effects applied across different runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to execute in the detailing process, affecting the depth and complexity of the applied effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration parameter for the detailing process, allowing for fine-tuning of the animation effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampler to be used in the detailing process, influencing the variation and quality of the applied effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduling algorithm for the detailing process, impacting the efficiency and outcome of the animation effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`denoise`**
    - The denoising factor applied during the detailing process, affecting the clarity and quality of the animation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`feather`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`basic_pipe`**
    - The basic pipeline configuration for the animation detailing process, defining the core components and their interactions.
    - Comfy dtype: `BASIC_PIPE`
    - Python dtype: `BasicPipe`
- **`refiner_ratio`**
    - The ratio of refinement applied to the detailing process, adjusting the intensity of the effects for a more nuanced outcome.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`detailer_hook`**
    - unknown
    - Comfy dtype: `DETAILER_HOOK`
    - Python dtype: `unknown`
- **`refiner_basic_pipe_opt`**
    - unknown
    - Comfy dtype: `BASIC_PIPE`
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
    - The enhanced image frames after applying animation effects, showcasing the improved visual impact.
    - Python dtype: `List[Image]`
- **`segs`**
    - Comfy dtype: `SEGS`
    - The updated segmentation information reflecting the detailed effects applied to specific areas of the animation.
    - Python dtype: `List[Segmentation]`
- **`basic_pipe`**
    - Comfy dtype: `BASIC_PIPE`
    - The basic pipeline configuration after the detailing process, potentially modified to incorporate new detailing components.
    - Python dtype: `BasicPipe`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - A collection of images generated during the detailing process, used for further refinement or as part of the animation.
    - Python dtype: `List[Image]`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [VHS_VideoCombine](../../ComfyUI-VideoHelperSuite/Nodes/VHS_VideoCombine.md)



## Source code
```python
class DetailerForEachPipeForAnimateDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                      "image_frames": ("IMAGE", ),
                      "segs": ("SEGS", ),
                      "guide_size": ("FLOAT", {"default": 512, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                      "max_size": ("FLOAT", {"default": 1024, "min": 64, "max": nodes.MAX_RESOLUTION, "step": 8}),
                      "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                      "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                      "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                      "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                      "scheduler": (core.SCHEDULERS,),
                      "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                      "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                      "basic_pipe": ("BASIC_PIPE", ),
                      "refiner_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                      },
                "optional": {
                      "detailer_hook": ("DETAILER_HOOK",),
                      "refiner_basic_pipe_opt": ("BASIC_PIPE",),
                      "noise_mask_feather": ("INT", {"default": 20, "min": 0, "max": 100, "step": 1}),
                      "scheduler_func_opt": ("SCHEDULER_FUNC",),
                      }
                }

    RETURN_TYPES = ("IMAGE", "SEGS", "BASIC_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "segs", "basic_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, False, False, True)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detailer"

    @staticmethod
    def doit(image_frames, segs, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
             denoise, feather, basic_pipe, refiner_ratio=None, detailer_hook=None, refiner_basic_pipe_opt=None,
             noise_mask_feather=0, scheduler_func_opt=None):

        enhanced_segs = []
        cnet_image_list = []

        for sub_seg in segs[1]:
            single_seg = segs[0], [sub_seg]
            enhanced_seg, cnet_images = SEGSDetailerForAnimateDiff().do_detail(image_frames, single_seg, guide_size, guide_size_for, max_size, seed, steps, cfg, sampler_name, scheduler,
                                                                               denoise, basic_pipe, refiner_ratio, refiner_basic_pipe_opt, noise_mask_feather, scheduler_func_opt=scheduler_func_opt)

            image_frames = SEGSPaste.doit(image_frames, enhanced_seg, feather, alpha=255)[0]

            if cnet_images is not None:
                cnet_image_list.extend(cnet_images)

            if detailer_hook is not None:
                image_frames = detailer_hook.post_paste(image_frames)

            enhanced_segs += enhanced_seg[1]

        new_segs = segs[0], enhanced_segs
        return image_frames, new_segs, basic_pipe, cnet_image_list

```
