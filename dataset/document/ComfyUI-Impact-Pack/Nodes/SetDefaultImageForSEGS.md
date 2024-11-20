---
tags:
- ImpactPack
- Segmentation
---

# Set Default Image for SEGS
## Documentation
- Class name: `SetDefaultImageForSEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node is designed to set a default image for SEGS (segmentation structures) in scenarios where the original image might be missing or not applicable. It ensures that each segment within the SEGS structure has a consistent and predefined visual representation, facilitating further processing or visualization steps without dependency on the original image data.
## Input types
### Required
- **`segs`**
    - The 'segs' input represents the segmentation structures for which a default image is to be set. This input is crucial for identifying the segments that require a default image assignment.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[SEG]`
- **`image`**
    - The 'image' input specifies the default image to be applied across all segments within the SEGS structure. It plays a key role in ensuring visual consistency across segments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`override`**
    - The 'override' input indicates whether the default image should replace existing images within the SEGS structure, allowing for flexible control over image assignment.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The output 'segs' consists of the updated segmentation structures with the default image applied, ready for further processing or visualization.
    - Python dtype: `List[SEG]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DefaultImageForSEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS", ),
                    "image": ("IMAGE", ),
                    "override": ("BOOLEAN", {"default": True}),
                }}

    RETURN_TYPES = ("SEGS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(segs, image, override):
        results = []

        segs = core.segs_scale_match(segs, image.shape)

        if len(segs[1]) > 0:
            if segs[1][0].cropped_image is not None:
                batch_count = len(segs[1][0].cropped_image)
            else:
                batch_count = len(image)

            for seg in segs[1]:
                if seg.cropped_image is not None and not override:
                    cropped_image = seg.cropped_image
                else:
                    cropped_image = None
                    for i in range(0, batch_count):
                        # take from original image
                        ref_image = image[i].unsqueeze(0)
                        cropped_image2 = crop_image(ref_image, seg.crop_region)

                        if cropped_image is None:
                            cropped_image = cropped_image2
                        else:
                            cropped_image = torch.cat((cropped_image, cropped_image2), dim=0)

                new_seg = SEG(cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
                results.append(new_seg)

            return ((segs[0], results), )
        else:
            return (segs, )

```
