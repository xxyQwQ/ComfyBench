---
tags:
- Mask
- MaskBatch
---

# 🔧 Mask Batch
## Documentation
- Class name: `MaskBatch+`
- Category: `essentials/mask batch`
- Output node: `False`

The MaskBatch+ node is designed for batch processing of mask images, specifically for combining two mask images into a single batch. It ensures compatibility between mask sizes through resizing operations if necessary, facilitating the integration of masks from different sources or dimensions into a unified batch format.
## Input types
### Required
- **`mask1`**
    - The first mask image to be combined into the batch. Its dimensions are checked against the second mask to ensure compatibility.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`mask2`**
    - The second mask image to be combined with the first. If its dimensions differ from the first mask, it is resized to match, ensuring uniformity in the batch.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The combined batch of the two input masks, returned as a single tensor. This facilitates further processing or analysis of the masks as a unified entity.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask batch"

    def execute(self, mask1, mask2):
        if mask1.shape[1:] != mask2.shape[1:]:
            mask2 = comfy.utils.common_upscale(mask2.unsqueeze(1).expand(-1,3,-1,-1), mask1.shape[2], mask1.shape[1], upscale_method='bicubic', crop='center')[:,0,:,:]

        return (torch.cat((mask1, mask2), dim=0),)

```
