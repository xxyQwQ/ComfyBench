---
tags:
- Counting
---

# 🔧 Batch Count
## Documentation
- Class name: `BatchCount+`
- Category: `essentials/utilities`
- Output node: `False`

The `BatchCount` node is designed to calculate the total number of elements within a given batch. It supports various data structures such as tensors, dictionaries, and lists, adapting its counting mechanism based on the structure's characteristics to accurately determine the batch size.
## Input types
### Required
- **`batch`**
    - The `batch` parameter is the primary input for the node, representing the data collection whose size is to be determined. It plays a crucial role in the node's functionality by allowing it to adapt its counting mechanism to the input's structure.
    - Comfy dtype: `*`
    - Python dtype: `Union[torch.Tensor, dict, list]`
## Output types
- **`int`**
    - Comfy dtype: `INT`
    - This output represents the total count of elements within the input batch, providing a single integer value as the result.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class BatchCount:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch": (any, {}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "execute"
    CATEGORY = "essentials/utilities"

    def execute(self, batch):
        count = 0
        if hasattr(batch, 'shape'):
            count = batch.shape[0]
        elif isinstance(batch, dict) and 'samples' in batch:
            count = batch['samples'].shape[0]
        elif isinstance(batch, list) or isinstance(batch, dict):
            count = len(batch)

        return (count, )

```
