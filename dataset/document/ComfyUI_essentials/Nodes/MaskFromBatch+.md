---
tags:
- Mask
- MaskBatch
---

# 🔧 Mask From Batch
## Documentation
- Class name: `MaskFromBatch+`
- Category: `essentials/mask batch`
- Output node: `False`

The MaskFromBatch+ node is designed for extracting a specific segment from a batch of masks based on the provided start index and length. It enables selective focus on particular portions of mask data for further processing or analysis.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the batch of masks from which a segment will be extracted. It is crucial for specifying the data subset to be operated on.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`start`**
    - The 'start' parameter determines the starting index from which the mask segment will be extracted, allowing for precise control over the selection of the data subset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`length`**
    - The 'length' parameter specifies the number of masks to be extracted from the batch, enabling the extraction of a specific range of data for focused analysis or processing.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - Returns a segment of the mask batch, extracted based on the specified start index and length, facilitating targeted manipulation or examination of mask data.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MaskFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": 1, "min": 1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask batch"

    def execute(self, mask, start, length):
        if length > mask.shape[0]:
            length = mask.shape[0]

        start = min(start, mask.shape[0]-1)
        length = min(mask.shape[0]-start, length)
        return (mask[start:start + length], )

```
