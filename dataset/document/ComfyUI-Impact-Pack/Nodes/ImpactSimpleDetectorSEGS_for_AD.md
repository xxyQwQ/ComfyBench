---
tags:
- Image
- Segmentation
---

# Simple Detector for AnimateDiff (SEGS)
## Documentation
- Class name: `ImpactSimpleDetectorSEGS_for_AD`
- Category: `ImpactPack/Detector`
- Output node: `False`

This node is designed to perform complex detection tasks tailored for AnimateDiff (AD) applications within the SEGS framework, utilizing a combination of bounding box and segmentation models to process elements in image sequences. It supports a variety of configurations to optimize detection for animated content creation or manipulation, by identifying key visual components across frames.
## Input types
### Required
- **`bbox_detector`**
    - Specifies the bounding box detector model used for initial detection steps, crucial for identifying preliminary regions of interest in the image frames.
    - Comfy dtype: `BBOX_DETECTOR`
    - Python dtype: `str`
- **`image_frames`**
    - A sequence of images on which detection is performed. These frames serve as the primary data source for the detection algorithm, enabling temporal analysis across frames.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`bbox_threshold`**
    - A threshold value for bounding box model predictions, controlling the sensitivity of detection. It influences the initial filtering of detected regions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bbox_dilation`**
    - Specifies the dilation amount for the bounding boxes detected, allowing for adjustments in the boundaries to better capture the relevant segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_factor`**
    - Determines the factor by which the detected bounding boxes are expanded or contracted, affecting the final cropping of segments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`drop_size`**
    - The minimum size of detected segments to be considered, helping to eliminate noise by disregarding overly small detections.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sub_threshold`**
    - A threshold value for sub-segmentation within the initially detected bounding boxes, refining the detection process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sub_dilation`**
    - Specifies the dilation amount for the sub-segmentation masks, fine-tuning the segmentation boundaries within each bounding box.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sub_bbox_expansion`**
    - Determines the expansion size for bounding boxes during the sub-segmentation process, allowing for more inclusive segment capture.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sam_mask_hint_threshold`**
    - A threshold value that influences the selection of segments for SAM (Segmentation-Aided Masking) model hints, guiding the model towards more relevant areas.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`masking_mode`**
    - Defines the strategy for combining detected segments across frames, affecting how animations are constructed from the detected elements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`segs_pivot`**
    - Specifies the reference frame or mask used for segment alignment and combination, pivotal in determining the base for further processing.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sam_model_opt`**
    - An optional parameter that specifies the SAM model to be used for enhanced segmentation, offering additional control over the detection quality.
    - Comfy dtype: `SAM_MODEL`
    - Python dtype: `str`
- **`segm_detector_opt`**
    - An optional parameter that specifies the segmentation detector model for refined detection, complementing the bounding box detection step.
    - Comfy dtype: `SEGM_DETECTOR`
    - Python dtype: `str`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output consists of segmented elements across the image frames, identified and processed based on the specified models and parameters. These segments are essential for the creation or manipulation of animated content in AnimateDiff applications.
    - Python dtype: `List[SEG]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [DetailerForEachPipeForAnimateDiff](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachPipeForAnimateDiff.md)
    - [ImpactControlNetApplySEGS](../../ComfyUI-Impact-Pack/Nodes/ImpactControlNetApplySEGS.md)



## Source code
```python
class SimpleDetectorForAnimateDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "bbox_detector": ("BBOX_DETECTOR", ),
                        "image_frames": ("IMAGE", ),

                        "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "bbox_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),

                        "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 100, "step": 0.1}),
                        "drop_size": ("INT", {"min": 1, "max": MAX_RESOLUTION, "step": 1, "default": 10}),

                        "sub_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                        "sub_dilation": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                        "sub_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),

                        "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                      },
                "optional": {
                        "masking_mode": (["Pivot SEGS", "Combine neighboring frames", "Don't combine"],),
                        "segs_pivot": (["Combined mask", "1st frame mask"],),
                        "sam_model_opt": ("SAM_MODEL", ),
                        "segm_detector_opt": ("SEGM_DETECTOR", ),
                 }
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Detector"

    @staticmethod
    def detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
               sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
               masking_mode="Pivot SEGS", segs_pivot="Combined mask", sam_model_opt=None, segm_detector_opt=None):

        h = image_frames.shape[1]
        w = image_frames.shape[2]

        # gather segs for all frames
        segs_by_frames = []
        for image in image_frames:
            image = image.unsqueeze(0)
            segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, crop_factor, drop_size)

            if sam_model_opt is not None:
                mask = core.make_sam_mask(sam_model_opt, segs, image, "center-1", sub_dilation,
                                          sub_threshold, sub_bbox_expansion, sam_mask_hint_threshold, False)
                segs = core.segs_bitwise_and_mask(segs, mask)
            elif segm_detector_opt is not None:
                segm_segs = segm_detector_opt.detect(image, sub_threshold, sub_dilation, crop_factor, drop_size)
                mask = core.segs_to_combined_mask(segm_segs)
                segs = core.segs_bitwise_and_mask(segs, mask)

            segs_by_frames.append(segs)

        def get_masked_frames():
            masks_by_frame = []
            for i, segs in enumerate(segs_by_frames):
                masks_in_frame = segs_nodes.SEGSToMaskList().doit(segs)[0]
                current_frame_mask = (masks_in_frame[0] * 255).to(torch.uint8)

                for mask in masks_in_frame[1:]:
                    current_frame_mask |= (mask * 255).to(torch.uint8)

                current_frame_mask = (current_frame_mask/255.0).to(torch.float32)
                current_frame_mask = utils.to_binary_mask(current_frame_mask, 0.1)[0]

                masks_by_frame.append(current_frame_mask)

            return masks_by_frame

        def get_empty_mask():
            return torch.zeros((h, w), dtype=torch.float32, device="cpu")

        def get_neighboring_mask_at(i, masks_by_frame):
            prv = masks_by_frame[i-1] if i > 1 else get_empty_mask()
            cur = masks_by_frame[i]
            nxt = masks_by_frame[i-1] if i > 1 else get_empty_mask()

            prv = prv if prv is not None else get_empty_mask()
            cur = cur.clone() if cur is not None else get_empty_mask()
            nxt = nxt if nxt is not None else get_empty_mask()

            return prv, cur, nxt

        def get_merged_neighboring_mask(masks_by_frame):
            if len(masks_by_frame) <= 1:
                return masks_by_frame

            result = []
            for i in range(0, len(masks_by_frame)):
                prv, cur, nxt = get_neighboring_mask_at(i, masks_by_frame)
                cur = (cur * 255).to(torch.uint8)
                cur |= (prv * 255).to(torch.uint8)
                cur |= (nxt * 255).to(torch.uint8)
                cur = (cur / 255.0).to(torch.float32)
                cur = utils.to_binary_mask(cur, 0.1)[0]
                result.append(cur)

            return result

        def get_whole_merged_mask():
            all_masks = []
            for segs in segs_by_frames:
                all_masks += segs_nodes.SEGSToMaskList().doit(segs)[0]

            merged_mask = (all_masks[0] * 255).to(torch.uint8)
            for mask in all_masks[1:]:
                merged_mask |= (mask * 255).to(torch.uint8)

            merged_mask = (merged_mask / 255.0).to(torch.float32)
            merged_mask = utils.to_binary_mask(merged_mask, 0.1)[0]
            return merged_mask

        def get_pivot_segs():
            if segs_pivot == "1st frame mask":
                return segs_by_frames[0][1]
            else:
                merged_mask = get_whole_merged_mask()
                return segs_nodes.MaskToSEGS.doit(merged_mask, False, crop_factor, False, drop_size, contour_fill=True)[0]

        def get_segs(merged_neighboring=False):
            pivot_segs = get_pivot_segs()

            masks_by_frame = get_masked_frames()
            if merged_neighboring:
                masks_by_frame = get_merged_neighboring_mask(masks_by_frame)

            new_segs = []
            for seg in pivot_segs[1]:
                cropped_mask = torch.zeros(seg.cropped_mask.shape, dtype=torch.float32, device="cpu").unsqueeze(0)
                pivot_mask = torch.from_numpy(seg.cropped_mask)
                x1, y1, x2, y2 = seg.crop_region
                for mask in masks_by_frame:
                    cropped_mask_at_frame = (mask[y1:y2, x1:x2] * pivot_mask).unsqueeze(0)
                    cropped_mask = torch.cat((cropped_mask, cropped_mask_at_frame), dim=0)

                if len(cropped_mask) > 1:
                    cropped_mask = cropped_mask[1:]

                new_seg = SEG(seg.cropped_image, cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                new_segs.append(new_seg)

            return pivot_segs[0], new_segs

        # create result mask
        if masking_mode == "Pivot SEGS":
            return (get_pivot_segs(), )

        elif masking_mode == "Combine neighboring frames":
            return (get_segs(merged_neighboring=True), )

        else: # elif masking_mode == "Don't combine":
            return (get_segs(merged_neighboring=False), )

    def doit(self, bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
             sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
             masking_mode="Pivot SEGS", segs_pivot="Combined mask", sam_model_opt=None, segm_detector_opt=None):

        return SimpleDetectorForAnimateDiff.detect(bbox_detector, image_frames, bbox_threshold, bbox_dilation, crop_factor, drop_size,
                                                   sub_threshold, sub_dilation, sub_bbox_expansion, sam_mask_hint_threshold,
                                                   masking_mode, segs_pivot, sam_model_opt, segm_detector_opt)

```
