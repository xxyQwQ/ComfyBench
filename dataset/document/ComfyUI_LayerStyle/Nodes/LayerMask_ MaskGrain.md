# LayerMask: Mask Grain
## Documentation
- Class name: `LayerMask: MaskGrain`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskGrain node applies a grain effect to a given mask, optionally inverting the mask and adjusting the grain intensity for a more textured appearance. It's designed to enhance or modify the visual texture of layer masks in image processing workflows, providing a tool for creative and detailed mask manipulation.
## Input types
### Required
- **`mask`**
    - The 'mask' parameter represents the input mask to which the grain effect will be applied. It is crucial for defining the area of the image that will be affected by the grain texture.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`grain`**
    - The 'grain' parameter determines the intensity of the grain effect applied to the mask. A higher value results in a more pronounced grain texture, allowing for customizable visual effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`invert_mask`**
    - The 'invert_mask' parameter allows for the inversion of the input mask before applying the grain effect, enabling flexibility in how the grain texture is applied to the image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The output is a modified mask with the grain effect applied, potentially inverted based on the input parameters. It's suitable for further image processing or as a final textured mask in creative projects.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskGrain:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "mask": ("MASK", ),  #
                "grain": ("INT", {"default": 6, "min": 0, "max": 127, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'mask_grain'
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_grain(self, mask, grain, invert_mask):

        l_masks = []
        ret_masks = []

        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for m in mask:
            if invert_mask:
                m = 1 - m
            l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for mask in l_masks:
            if grain:
                white_mask = Image.new('L', mask.size, color="white")
                inner_mask = tensor2pil(expand_mask(image2mask(mask), 0 - grain, int(grain))).convert('L')
                outter_mask = tensor2pil(expand_mask(image2mask(mask), grain, int(grain * 2))).convert('L')
                ret_mask = Image.new('L', mask.size, color="black")
                ret_mask = chop_image_v2(ret_mask, outter_mask, blend_mode="dissolve", opacity=50).convert('L')
                ret_mask.paste(white_mask, mask=inner_mask)
                ret_masks.append(image2mask(ret_mask))
            else:
                ret_masks.append(image2mask(mask))

        log(f"{NODE_NAME} Processed {len(ret_masks)} mask(s).", message_type='finish')
        return (torch.cat(ret_masks, dim=0),)

```
