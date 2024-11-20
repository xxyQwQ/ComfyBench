---
tags:
- Mask
- MaskGeneration
---

# LayerMask: Shadow & Highlight Mask
## Documentation
- Class name: `LayerMask: Shadow & Highlight Mask`
- Category: `😺dzNodes/LayerMask`
- Output node: `False`

This node is designed to generate shadow and highlight masks for images, allowing for detailed manipulation of lighting effects in image processing tasks. It processes images to extract or enhance shadows and highlights based on specified level offsets and ranges, supporting both aesthetic adjustments and functional analysis.
## Input types
### Required
- **`image`**
    - The primary image input for which shadow and highlight masks are to be generated. It plays a crucial role in determining the areas of shadow and highlight to be enhanced or extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`shadow_level_offset`**
    - Specifies the offset for shadow levels, adjusting the intensity of shadows in the image. This parameter fine-tunes the shadow mask generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_range`**
    - Defines the range of shadow levels to be considered, affecting the breadth of shadows captured in the mask. It helps in customizing the depth and extent of shadows.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`highlight_level_offset`**
    - Specifies the offset for highlight levels, adjusting the brightness of highlights in the image. This parameter fine-tunes the highlight mask generation process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`highlight_range`**
    - Defines the range of highlight levels to be considered, affecting the breadth of highlights captured in the mask. It helps in customizing the intensity and extent of highlights.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask`**
    - An optional mask that can be applied to the image to focus the shadow and highlight mask generation on specific areas, enhancing processing efficiency and precision.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`shadow_mask`**
    - Comfy dtype: `MASK`
    - The generated mask that emphasizes shadows in the image, useful for further image manipulation or analysis.
    - Python dtype: `torch.Tensor`
- **`highlight_mask`**
    - Comfy dtype: `MASK`
    - The generated mask that emphasizes highlights in the image, useful for further image manipulation or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ShadowAndHighlightMask:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "shadow_level_offset": ("INT", {"default": 0, "min": -99, "max": 99, "step": 1}),
                "shadow_range": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
                "highlight_level_offset": ("INT", {"default": 0, "min": -99, "max": 99, "step": 1}),
                "highlight_range": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("shadow_mask", "highlight_mask")
    FUNCTION = 'shadow_and_highlight_mask'
    CATEGORY = '😺dzNodes/LayerMask'

    def shadow_and_highlight_mask(self, image,
                                  shadow_level_offset, shadow_range,
                                  highlight_level_offset, highlight_range,
                                  mask=None
                                  ):

        ret_shadow_masks = []
        ret_highlight_masks = []
        input_images = []
        input_masks = []

        for i in image:
            input_images.append(torch.unsqueeze(i, 0))
            m = tensor2pil(i)
            if m.mode == 'RGBA':
                input_masks.append(m.split()[-1])
            else:
                input_masks.append(Image.new('L', size=m.size, color='white'))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            input_masks = []
            for m in mask:
                input_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(input_images), len(input_masks))

        for i in range(max_batch):
            _image = input_images[i] if i < len(input_images) else input_images[-1]
            _image = tensor2pil(_image).convert('RGB')
            _mask = input_masks[i] if i < len(input_masks) else input_masks[-1]


            avg_gray = get_gray_average(_image, _mask)
            shadow_level, highlight_level = calculate_shadow_highlight_level(avg_gray)
            shadow_low_threshold = (shadow_level + shadow_level_offset) / 100 + shadow_range / 2
            shadow_low_threshold = norm_value(shadow_low_threshold)
            shadow_high_threshold = (shadow_level + shadow_level_offset) / 100 - shadow_range / 2
            shadow_high_threshold = norm_value(shadow_high_threshold)
            _shadow_mask = luminance_keyer(_image, shadow_low_threshold, shadow_high_threshold)

            highlight_low_threshold = (highlight_level + highlight_level_offset) / 100 - highlight_range / 2
            highlight_low_threshold = norm_value(highlight_low_threshold)
            highlight_high_threshold = (highlight_level + highlight_level_offset) / 100 + highlight_range / 2
            highlight_high_threshold = norm_value(highlight_high_threshold)
            _highlight_mask = luminance_keyer(_image, highlight_low_threshold, highlight_high_threshold)

            black = Image.new('L', size=_image.size, color='black')
            _mask = ImageChops.invert(_mask)
            _shadow_mask.paste(black, mask=_mask)
            _highlight_mask.paste(black, mask=_mask)
            ret_shadow_masks.append(image2mask(_shadow_mask))
            ret_highlight_masks.append(image2mask(_highlight_mask))

        log(f"{NODE_NAME} Processed {len(ret_shadow_masks)} image(s).", message_type='finish')
        return (torch.cat(ret_shadow_masks, dim=0),torch.cat(ret_highlight_masks, dim=0),)

```
