---
tags:
- DetailEnhancement
- Image
- Pipeline
---

# FaceDetailerPipe (AV)
## Documentation
- Class name: `AV_FaceDetailerPipe`
- Category: `ArtVenture/Detailer`
- Output node: `False`

The AV_FaceDetailerPipe node enhances facial details in images within the ArtVenture Detailer category, leveraging an optional enabled flag to control its operation. It builds upon the FaceDetailerPipe's functionality, allowing for detailed customization and refinement of facial features in artwork or photographs.
## Input types
### Required
- **`image`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`detailer_pipe`**
    - unknown
    - Comfy dtype: `DETAILER_PIPE`
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
- **`bbox_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`bbox_dilation`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`bbox_crop_factor`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_detection_hint`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`sam_dilation`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`sam_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_bbox_expansion`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`sam_mask_hint_threshold`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`sam_mask_hint_use_negative`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`drop_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`refiner_ratio`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`cycle`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
### Optional
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
- **`enabled`**
    - A boolean flag that determines whether the face detailing process is activated. When enabled, the node performs facial detailing on the input image; when disabled, it bypasses the detailing process and returns the original image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image, which may be the original or a version with enhanced facial details, depending on whether the detailing process was enabled.
    - Python dtype: `Image`
- **`cropped_refined`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`cropped_enhanced_alpha`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`mask`**
    - Comfy dtype: `MASK`
    - unknown
    - Python dtype: `unknown`
- **`detailer_pipe`**
    - Comfy dtype: `DETAILER_PIPE`
    - The detailer pipe configuration used for processing the image. This output is part of the node's mechanism to allow further customization or analysis.
    - Python dtype: `DetailerPipe`
- **`cnet_images`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
    class AV_FaceDetailerPipe(FaceDetailerPipe):
        @classmethod
        def INPUT_TYPES(s):
            inputs = FaceDetailerPipe.INPUT_TYPES()
            inputs["optional"]["enabled"] = (
                "BOOLEAN",
                {"default": True, "label_on": "enabled", "label_off": "disabled"},
            )
            return inputs

        CATEGORY = "ArtVenture/Detailer"

        def doit(self, image, detailer_pipe, *args, enabled=True, **kwargs):
            if enabled:
                return super().doit(image, detailer_pipe, *args, **kwargs)
            else:
                return (image, [], [], None, detailer_pipe, [])

```
