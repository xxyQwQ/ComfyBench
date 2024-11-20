---
tags:
- Mask
- MaskGeneration
---

# LayerMask: MaskGrow
## Documentation
- Class name: `LayerMask: MaskGrow`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskGrow node is designed to modify image masks by inverting, growing, and blurring them to achieve a desired visual effect. This node allows for the dynamic adjustment of mask boundaries and texture, facilitating more nuanced and flexible image manipulation.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the image mask to be processed. It is the primary input for transformations, determining the base form on which the node operates.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - The 'invert_mask' parameter controls whether the input mask should be inverted before processing. This inversion can be crucial for achieving specific visual effects or mask manipulations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`grow`**
    - The 'grow' parameter specifies the amount by which the mask's boundaries should be expanded or contracted, allowing for dynamic resizing of the mask area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - The 'blur' parameter determines the intensity of the blur effect applied to the mask's edges, enabling the creation of softer or more pronounced boundaries.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output 'mask' is the transformed version of the input mask, after applying the specified inversion, growth, and blur operations.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskGrow:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "grow": ("INT", {"default": 4, "min": -999, "max": 999, "step": 1}),
                "blur": ("INT", {"default": 4, "min": 0, "max": 999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_grow'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_grow(self, mask, invert_mask, grow, blur,):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)

        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_masks)):

            _mask = l_masks[i]
            ret_masks.append(expand_mask(image2mask(_mask), grow, blur) )

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
