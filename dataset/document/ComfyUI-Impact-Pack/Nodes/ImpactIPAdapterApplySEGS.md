---
tags:
- IPAdapter
---

# IPAdapterApply (SEGS)
## Documentation
- Class name: `ImpactIPAdapterApplySEGS`
- Category: `ImpactPack/Util`
- Output node: `False`

This node applies an IP Adapter to SEGS (segmentation elements), enhancing or modifying them based on a set of parameters and a reference image. It allows for the dynamic adjustment of segmentation elements through image processing techniques and neural network models, facilitating complex image manipulation tasks.
## Input types
### Required
- **`segs`**
    - The segmentation elements to be processed, typically including information such as crop regions and masks.
    - Comfy dtype: `SEGS`
    - Python dtype: `List[Tuple[Any, List[SEG]]]`
- **`ipadapter_pipe`**
    - A pipeline of models or functions that the IP Adapter will use to process the segmentation elements.
    - Comfy dtype: `IPADAPTER_PIPE`
    - Python dtype: `List[Callable]`
- **`weight`**
    - A parameter controlling the influence of the IP Adapter on the segmentation elements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise`**
    - An optional noise parameter that can be applied for variability in the adaptation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the type of weighting mechanism used in the adaptation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Defines the starting point within the pipeline for applying the IP Adapter.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`end_at`**
    - Specifies the endpoint within the pipeline for the application of the IP Adapter.
    - Comfy dtype: `FLOAT`
    - Python dtype: `int`
- **`unfold_batch`**
    - Determines whether the processing should unfold in batches for efficiency.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`faceid_v2`**
    - An optional parameter for face identification, relevant in certain contexts.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`weight_v2`**
    - A secondary weight parameter, potentially used for more nuanced control.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`context_crop_factor`**
    - Adjusts the crop region in relation to the context, affecting how much of the surrounding area is considered in the adaptation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`reference_image`**
    - The reference image against which the segmentation elements are adapted or modified.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`combine_embeds`**
    - Determines how embeddings are combined in the adaptation process, with options such as 'concat'.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`neg_image`**
    - An optional negative image that can be used for contrast or as a counter-reference in the adaptation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Optional[torch.Tensor]`
## Output types
- **`segs`**
    - Comfy dtype: `SEGS`
    - The enhanced or modified segmentation elements after applying the IP Adapter.
    - Python dtype: `Tuple[Tuple[Any, List[SEG]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterApplySEGS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "segs": ("SEGS",),
                    "ipadapter_pipe": ("IPADAPTER_PIPE",),
                    "weight": ("FLOAT", {"default": 0.7, "min": -1, "max": 3, "step": 0.05}),
                    "noise": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "weight_type": (["original", "linear", "channel penalty"], {"default": 'channel penalty'}),
                    "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "end_at": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.001}),
                    "unfold_batch": ("BOOLEAN", {"default": False}),
                    "faceid_v2": ("BOOLEAN", {"default": False}),
                    "weight_v2": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                    "context_crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100, "step": 0.1}),
                    "reference_image": ("IMAGE",),
                    },
                "optional": {
                    "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                    "neg_image": ("IMAGE",),
                    },
                }

    RETURN_TYPES = ("SEGS",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(segs, ipadapter_pipe, weight, noise, weight_type, start_at, end_at, unfold_batch, faceid_v2, weight_v2, context_crop_factor, reference_image, combine_embeds="concat", neg_image=None):

        if len(ipadapter_pipe) == 4:
            print(f"[Impact Pack] IPAdapterApplySEGS: Installed Inspire Pack is outdated.")
            raise Exception("Inspire Pack is outdated.")

        new_segs = []

        h, w = segs[0]

        if reference_image.shape[2] != w or reference_image.shape[1] != h:
            reference_image = tensor_resize(reference_image, w, h)
        
        for seg in segs[1]:
            # The context_crop_region sets how much wider the IPAdapter context will reflect compared to the crop_region, not the bbox
            context_crop_region = make_crop_region(w, h, seg.crop_region, context_crop_factor)
            cropped_image = crop_image(reference_image, context_crop_region)

            control_net_wrapper = core.IPAdapterWrapper(ipadapter_pipe, weight, noise, weight_type, start_at, end_at, unfold_batch, weight_v2, cropped_image, neg_image=neg_image, prev_control_net=seg.control_net_wrapper, combine_embeds=combine_embeds)
            new_seg = SEG(seg.cropped_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, control_net_wrapper)
            new_segs.append(new_seg)

        return ((segs[0], new_segs), )

```
