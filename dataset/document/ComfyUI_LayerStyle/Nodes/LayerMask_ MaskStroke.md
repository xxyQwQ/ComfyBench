---
tags:
- Mask
---

# LayerMask: MaskStroke
## Documentation
- Class name: `LayerMask: MaskStroke`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskStroke node is designed to process image masks by applying transformations such as inversion, growth, width adjustment, and blurring to create stylized stroke effects around the mask's edges. It's aimed at enhancing visual elements by dynamically modifying mask boundaries for artistic or design purposes.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input image mask to be transformed. It is crucial for defining the area where stroke effects will be applied, serving as the base for subsequent modifications.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`invert_mask`**
    - The 'invert_mask' parameter allows for the inversion of the input mask, flipping the areas to be considered as the foreground and background. This inversion is essential for achieving desired visual effects based on the mask's context.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`stroke_grow`**
    - Specifies the growth or shrinkage of the stroke's inner boundary, adjusting the mask's coverage area before applying the stroke width. It's key for fine-tuning the effect's intensity and reach.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`stroke_width`**
    - Determines the thickness of the stroke applied around the mask's edges, directly influencing the visual prominence of the stroke effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blur`**
    - Controls the level of blur applied to the stroke, smoothing the edges for a more refined or subtle effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The transformed mask with the applied stroke effect, showcasing the adjustments made to the original mask through inversion, growth, width modification, and blurring.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskStroke:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
                "stroke_grow": ("INT", {"default": 0, "min": -999, "max": 999, "step": 1}),  # 收缩值
                "stroke_width": ("INT", {"default": 20, "min": 0, "max": 999, "step": 1}),  # 扩张值
                "blur": ("INT", {"default": 6, "min": 0, "max": 100, "step": 1}),  # 模糊
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_stroke'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_stroke(self, mask, invert_mask, stroke_grow, stroke_width, blur,):

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
            grow_offset = int(stroke_width / 2)
            inner_stroke = stroke_grow - grow_offset
            outer_stroke = inner_stroke + stroke_width
            inner_mask = expand_mask(image2mask(_mask), inner_stroke, blur)
            outer_mask = expand_mask(image2mask(_mask), outer_stroke, blur)
            stroke_mask = subtract_mask(outer_mask, inner_mask)
            ret_masks.append(stroke_mask)

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
