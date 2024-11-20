# Select Nth Mask (Inspire)
## Documentation
- Class name: `SelectNthMask __Inspire`
- Category: `InspirePack/Util`
- Output node: `False`

The SelectNthMask node is designed to select a specific mask from a batch of masks based on a given index. It abstracts the functionality of indexing into a collection of masks and retrieving a single mask, which can be particularly useful in workflows that require manipulation or analysis of individual masks within a larger dataset.
## Input types
### Required
- **`masks`**
    - The collection of masks from which a specific mask is to be selected. It plays a crucial role in determining the output by specifying the batch of available masks.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`idx`**
    - The index of the mask to be selected from the batch. This parameter dictates which mask is retrieved, allowing for targeted manipulation or analysis within a sequence of masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask selected from the input batch based on the specified index. It represents a single mask extracted for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SelectNthMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "masks": ("MASK",),
                    "idx": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                    },
                }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    CATEGORY = "InspirePack/Util"

    def doit(self, masks, idx):
        return (masks[idx].unsqueeze(0),)

```
