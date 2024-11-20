---
tags:
- ControlNet
---

# SparseCtrl Index Method 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlIndexMethodNode`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl`
- Output node: `False`

This node is designed to generate a sparse control method based on specified indexes. It abstracts the complexity of selecting specific indexes for sparse control in advanced control networks, facilitating targeted manipulation or analysis of data points within a given dataset.
## Input types
### Required
- **`indexes`**
    - Specifies the indexes to be used for the sparse control method. This parameter allows for targeted selection within the dataset, enabling precise control over which data points are manipulated or analyzed.
    - Comfy dtype: `STRING`
    - Python dtype: `list[int]`
## Output types
- **`sparse_method`**
    - Comfy dtype: `SPARSE_METHOD`
    - Returns a sparse method object configured with the specified indexes. This object is used to apply sparse control techniques within advanced control networks.
    - Python dtype: `SparseIndexMethod`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SparseIndexMethodNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "indexes": ("STRING", {"default": "0"}),
            }
        }
    
    RETURN_TYPES = ("SPARSE_METHOD",)
    FUNCTION = "get_method"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def get_method(self, indexes: str):
        idxs = get_idx_list_from_str(indexes)
        return (SparseIndexMethod(idxs),)

```
