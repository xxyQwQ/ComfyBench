---
tags:
- ImpactPack
- Segmentation
---

# Picker (SEGS)
## Documentation
- Class name: `ImpactSEGSPicker`
- Category: `ImpactPack/Util`
- Output node: `True`

The ImpactSEGSPicker node is designed for selecting specific segments from a collection based on user-defined criteria. It facilitates the generation of candidate images from the selected segments, enhancing the flexibility in processing and visualizing segment data.
## Input types
### Required
- **`picks`**
    - A comma-separated list of indices specifying which segments to pick from the collection. This parameter allows users to selectively process and output only the segments of interest.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`segs`**
    - The collection of segments from which specific items are to be selected. This parameter is central to the node's operation, serving as the input data that will be filtered according to the 'picks' parameter.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[Segment]`
### Optional
- **`fallback_image_opt`**
    - An optional image used to adjust the scale of segments to match its dimensions. This parameter is useful for ensuring consistency in segment sizes when necessary.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[Image]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The modified collection of segments after applying the selection and scaling operations, ready for further processing or visualization.
    - Python dtype: `Tuple[List[Segment]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SEGSPicker:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "picks": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                    "segs": ("SEGS",),
                    },
                "optional": {
                     "fallback_image_opt": ("IMAGE", ),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"},
                }

    RETURN_TYPES = ("SEGS", )

    OUTPUT_NODE = True

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(picks, segs, fallback_image_opt=None, unique_id=None):
        if fallback_image_opt is not None:
            segs = core.segs_scale_match(segs, fallback_image_opt.shape)

        # generate candidates image
        cands = []
        for seg in segs[1]:
            if seg.cropped_image is not None:
                cropped_image = seg.cropped_image
            elif fallback_image_opt is not None:
                # take from original image
                cropped_image = crop_image(fallback_image_opt, seg.crop_region)
            else:
                cropped_image = empty_pil_tensor()

            mask_array = seg.cropped_mask.copy()
            mask_array[mask_array < 0.3] = 0.3
            mask_array = mask_array[None, ..., None]
            cropped_image = cropped_image * mask_array

            cands.append(cropped_image)

        impact.impact_server.segs_picker_map[unique_id] = cands

        # pass only selected
        pick_ids = set()

        for pick in picks.split(","):
            try:
                pick_ids.add(int(pick)-1)
            except Exception:
                pass

        new_segs = []
        for i in pick_ids:
            if 0 <= i < len(segs[1]):
                new_segs.append(segs[1][i])

        return ((segs[0], new_segs),)

```
