---
tags:
- Mask
---

# Select Every Nth Mask 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthMask`
- Category: `Video Helper Suite 🎥🅥🅗🅢/mask`
- Output node: `False`

The node is designed to filter a batch of masks by selecting every Nth mask from the sequence, optionally skipping a specified number of initial masks. This functionality is useful for thinning out dense sequences of masks to reduce computational load or to select masks at a regular interval for processing or analysis.
## Input types
### Required
- **`mask`**
    - The input tensor containing a batch of masks from which the node will select every Nth mask. This parameter is crucial for defining the subset of masks to be processed.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
- **`select_every_nth`**
    - Specifies the interval at which masks are selected from the input batch. A higher value thins out the sequence more by selecting masks less frequently.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_masks`**
    - Determines the number of initial masks to skip before starting to select every Nth mask. This allows for the exclusion of a certain number of masks from the beginning of the sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`MASK`**
    - Comfy dtype: `MASK`
    - The output tensor containing the selected masks after applying the specified interval and skip criteria.
    - Python dtype: `Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of masks selected and returned by the node.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_masks": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/mask"

    RETURN_TYPES = ("MASK", "INT",)
    RETURN_NAMES = ("MASK", "count",)
    FUNCTION = "select_masks"

    def select_masks(self, mask: Tensor, select_every_nth: int, skip_first_masks: int):
        sub_mask = mask[skip_first_masks::select_every_nth]
        return (sub_mask, sub_mask.size(0))

```
