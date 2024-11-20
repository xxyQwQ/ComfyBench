# LayerMask: MaskInvert
## Documentation
- Class name: `LayerMask: MaskInvert`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskInvert node is designed to invert the values of a given mask, transforming white areas to black and vice versa. This operation is essential for tasks that require the opposite representation of mask areas, such as highlighting regions of interest in image processing or computer vision applications.
## Input types
### Required
- **`mask`**
    - The 'mask' input is the primary data the node operates on, representing the areas to be inverted. It plays a crucial role in determining the output, as the inversion process directly transforms this mask.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output 'mask' is the result of inverting the input mask, with all originally white areas turned black and vice versa, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskInvert:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "mask": ("MASK", ),  #
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_invert'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_invert(self,mask):
        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):
            _mask = l_masks[i]
            ret_masks.append(mask_invert(image2mask(_mask)))

        return (torch.cat(ret_masks, dim=0),)

```
