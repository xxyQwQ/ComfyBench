---
tags:
- Mask
- MaskGeneration
---

# Round Mask
## Documentation
- Class name: `RoundMask`
- Category: `KJNodes/masking`
- Output node: `False`

The RoundMask node specializes in converting masks or batches of masks into binary masks by rounding their values. This operation is essential for tasks that require clear, binary distinctions within masks, such as separating foreground from background in image processing applications.
## Input types
### Required
- **`mask`**
    - The mask input is the primary data that the RoundMask node processes. It rounds the values of this mask to produce a binary mask, which is crucial for applications needing distinct segmentation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a binary mask or a batch of binary masks, where each pixel value is rounded to the nearest integer, effectively converting the mask into a binary format.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RoundMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "mask": ("MASK",),  
        }}

    RETURN_TYPES = ("MASK",)
    FUNCTION = "round"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Rounds the mask or batch of masks to a binary mask.  
<img src="https://github.com/kijai/ComfyUI-KJNodes/assets/40791699/52c85202-f74e-4b96-9dac-c8bda5ddcc40" width="300" height="250" alt="RoundMask example">

"""

    def round(self, mask):
        mask = mask.round()
        return (mask,)

```
