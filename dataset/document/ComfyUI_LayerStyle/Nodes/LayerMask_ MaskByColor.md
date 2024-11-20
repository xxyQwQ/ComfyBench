---
tags:
- Mask
- MaskGeneration
---

# LayerMask: Mask by Color
## Documentation
- Class name: `LayerMask: MaskByColor`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

The MaskByColor node is designed to generate masks based on specific color criteria within images. It allows for the creation of layer masks by isolating areas that match a given color, with options to adjust the threshold for color matching, invert the mask, and apply gap fixing to improve mask quality.
## Input types
### Required
- **`image`**
    - The input image on which the mask generation is based. This image is analyzed to identify areas that match the specified color criteria.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`color`**
    - Specifies the target color for mask generation. This parameter is crucial for determining which areas of the image will be isolated to create the mask.
    - Comfy dtype: `COLOR`
    - Python dtype: `str`
- **`color_in_HEX`**
    - A hexadecimal string representing the target color for mask generation. This provides an alternative method for specifying the color, offering flexibility in defining the color criteria.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`threshold`**
    - Determines the sensitivity of color matching. A lower threshold results in a more inclusive selection of matching colors, while a higher threshold restricts the match to colors more closely resembling the specified target color.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_gap`**
    - Activates gap fixing in the mask to close small holes or gaps. This parameter can enhance the mask's quality by ensuring more continuous and coherent masked areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`fix_threshold`**
    - Sets the threshold for gap fixing, controlling the size of gaps to be filled. This parameter works in conjunction with 'fix_gap' to refine the mask's details and overall appearance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`invert_mask`**
    - When enabled, inverts the generated mask, swapping the masked and unmasked areas. This option allows for flexibility in selecting either the matching areas or their complements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`mask`**
    - unknown
    - Comfy dtype: `MASK`
    - Python dtype: `unknown`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - Returns a tensor of the generated mask, isolating areas matching the specified color criteria, adjusted for threshold, inversion, and gap fixing settings.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskByColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "color": ("COLOR", {"default": "#FFFFFF"},),
                "color_in_HEX": ("STRING", {"default": ""}),
                "threshold": ("INT", { "default": 0, "min": 0, "max": 100, "step": 1, }),
                "fix_gap": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "fix_threshold": ("FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_by_color"
    CATEGORY = '😺dzNodes/LayerMask'

    def mask_by_color(self, image, color, color_in_HEX, threshold,
                      fix_gap, fix_threshold, invert_mask, mask=None):

        if color_in_HEX != "" and color_in_HEX.startswith('#') and len(color_in_HEX) == 7:
            color = color_in_HEX

        ret_masks = []
        l_images = []
        l_masks = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        for i in range(len(l_images)):
            img = l_images[i] if i < len(l_images) else l_images[-1]
            img = tensor2pil(img)
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            mask = Image.new('L', _mask.size, 'black')
            mask.paste(create_mask_from_color_tensor(img, color, threshold), mask=_mask)
            mask = image2mask(mask)
            if invert_mask:
                mask = 1 - mask
            if fix_gap:
                mask = mask_fix(mask, 1, fix_gap, fix_threshold, fix_threshold)
            ret_masks.append(mask)

        return (torch.cat(ret_masks, dim=0), )

```
