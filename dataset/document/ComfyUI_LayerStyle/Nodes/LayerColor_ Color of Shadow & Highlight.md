---
tags:
- Color
---

# LayerColor: Color of Shadow & Highlight
## Documentation
- Class name: `LayerColor: Color of Shadow & Highlight`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

This node is designed to adjust the color properties of shadows and highlights within an image. It allows for the fine-tuning of brightness, saturation, and hue for both shadows and highlights, enabling users to achieve a desired visual effect or correct color imbalances.
## Input types
### Required
- **`image`**
    - The input image to be processed for shadow and highlight color correction.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`shadow_brightness`**
    - Specifies the brightness level for shadows, affecting the overall darkness or lightness of shadow areas in the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`shadow_saturation`**
    - Determines the saturation level for shadows, influencing the intensity of colors in shadowed regions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`shadow_hue`**
    - Adjusts the hue for shadows, allowing for the alteration of color tones in shadow areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_level_offset`**
    - Offsets the level of shadows, modifying the threshold at which shadows are defined.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`shadow_range`**
    - Defines the range of shadow levels, affecting the breadth of tones considered as shadows.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`highlight_brightness`**
    - Defines the brightness level for highlights, impacting the luminosity of highlighted areas in the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`highlight_saturation`**
    - Sets the saturation level for highlights, affecting the vibrancy of colors in highlighted regions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`highlight_hue`**
    - Modifies the hue for highlights, enabling changes in color tones within highlighted areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`highlight_level_offset`**
    - Offsets the level of highlights, modifying the threshold at which highlights are defined.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`highlight_range`**
    - Defines the range of highlight levels, affecting the breadth of tones considered as highlights.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask`**
    - An optional mask to specify areas of the image to apply the shadow and highlight adjustments.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying shadow and highlight color corrections.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectShadowAndHighlight:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),
                "shadow_brightness": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "shadow_saturation": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "shadow_hue": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "shadow_level_offset": ("INT", {"default": 0, "min": -99, "max": 99, "step": 1}),
                "shadow_range": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
                "highlight_brightness": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "highlight_saturation": ("FLOAT", {"default": 1, "min": 0.0, "max": 3, "step": 0.01}),
                "highlight_hue": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "highlight_level_offset": ("INT", {"default": 0, "min": -99, "max": 99, "step": 1}),
                "highlight_range": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.99, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_shadow_and_highlight'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_shadow_and_highlight(self, image,
                               shadow_brightness, shadow_saturation,
                               shadow_level_offset, shadow_range, shadow_hue,
                               highlight_brightness, highlight_saturation, highlight_hue,
                               highlight_level_offset, highlight_range,
                               mask=None
                               ):

        ret_images = []
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
            _canvas = _image.copy()
            if shadow_saturation !=1 or shadow_brightness !=1 or shadow_hue:
                shadow_low_threshold = (shadow_level + shadow_level_offset) / 100 + shadow_range / 2
                shadow_low_threshold = norm_value(shadow_low_threshold)
                shadow_high_threshold = (shadow_level + shadow_level_offset) / 100 - shadow_range / 2
                shadow_high_threshold = norm_value(shadow_high_threshold)
                _shadow_mask = luminance_keyer(_image, shadow_low_threshold, shadow_high_threshold)
                _shadow = _image.copy()
                if shadow_brightness != 1:
                    brightness_image = ImageEnhance.Brightness(_shadow)
                    _shadow = brightness_image.enhance(factor=shadow_brightness)
                if shadow_saturation != 1:
                    color_image = ImageEnhance.Color(_shadow)
                    _shadow = color_image.enhance(factor=shadow_saturation)
                if shadow_hue:
                    _h, _s, _v = _shadow.convert('HSV').split()
                    _h = image_hue_offset(_h, shadow_hue)
                    _shadow = image_channel_merge((_h, _s, _v), 'HSV')
                _canvas.paste(_shadow, mask=gaussian_blur(_shadow_mask,(_shadow_mask.width + _shadow_mask.height)//800))
                _canvas.paste(_image, mask=ImageChops.invert(_mask))
            if highlight_saturation != 1 or highlight_brightness != 1 or highlight_hue:
                highlight_low_threshold = (highlight_level + highlight_level_offset) / 100 - highlight_range / 2
                highlight_low_threshold = norm_value(highlight_low_threshold)
                highlight_high_threshold = (highlight_level + highlight_level_offset) / 100 + highlight_range / 2
                highlight_high_threshold = norm_value(highlight_high_threshold)
                _highlight_mask = luminance_keyer(_image, highlight_low_threshold, highlight_high_threshold)
                _highlight = _image.copy()
                if highlight_brightness != 1:
                    brightness_image = ImageEnhance.Brightness(_highlight)
                    _highlight = brightness_image.enhance(factor=highlight_brightness)
                if highlight_saturation != 1:
                    color_image = ImageEnhance.Color(_highlight)
                    _highlight = color_image.enhance(factor=highlight_saturation)
                if highlight_hue:
                    _h, _s, _v = _highlight.convert('HSV').split()
                    _h = image_hue_offset(_h, highlight_hue)
                    _highlight = image_channel_merge((_h, _s, _v), 'HSV')
                _canvas.paste(_highlight, mask=gaussian_blur(_highlight_mask, (_highlight_mask.width + _highlight_mask.height)//800))
                _canvas.paste(_image, mask=ImageChops.invert(_mask))
            ret_images.append(pil2tensor(_canvas))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
