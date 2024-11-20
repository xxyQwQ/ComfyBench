---
tags:
- Mask
- MaskBatch
---

# Mask Batch Manager (SuperBeasts.AI)
## Documentation
- Class name: `Mask Batch Manager (SuperBeasts.AI)`
- Category: `SuperBeastsAI/Masks`
- Output node: `False`

The Mask Batch Manager node is designed to aggregate multiple individual masks or batches of masks into a single, processed batch. It adjusts the size and order of the input masks according to specified dimensions and optional ordering, ensuring the output is a unified batch suitable for further processing or analysis.
## Input types
### Required
- **`width`**
    - Specifies the desired width of the output masks, affecting how each input mask is resized or cropped to match this width.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Determines the desired height of the output masks, influencing the resizing and cropping operations to ensure each mask conforms to this height.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`new_order`**
    - An optional parameter that allows specifying the order in which input masks should be arranged in the output batch, enabling custom sequencing of masks.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The aggregated output of processed masks, combined into a single batch for further use.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBatchManagement:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512}),
                "height": ("INT", {"default": 768})
            },
            "optional": {
                "new_order": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "append"
    CATEGORY = "SuperBeastsAI/Masks"

    def append(self, width, height, new_order, **kwargs):
        masks = [kwargs["mask1"]]  # Start with the required mask1 input

        i = 2
        while f"mask{i}" in kwargs:
            masks.append(kwargs[f"mask{i}"])
            i += 1

        if new_order:
            order_indices = [int(idx) - 1 for idx in new_order.split(',') if idx.strip()]
            masks = [masks[idx] for idx in order_indices if idx < len(masks)]

        processed_masks = []
        for mask in masks:
            pil_mask = tensor2pil(mask)
            resized_cropped_mask = resize_and_crop(pil_mask, width, height)
            mask_tensor = pil2tensor(resized_cropped_mask)
            processed_masks.append(mask_tensor)

        result = torch.cat(processed_masks, dim=0) if processed_masks else torch.empty(0, 1, height, width)

        return (result,)

```
