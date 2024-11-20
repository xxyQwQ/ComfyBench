---
tags:
- Color
---

# LayerColor: AutoAdjust
## Documentation
- Class name: `LayerColor: AutoAdjust`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The AutoAdjust node provides automated adjustments to the color properties of an image, including brightness, contrast, saturation, and individual color channels. It aims to enhance the image's overall appearance or correct its color balance based on specified parameters.
## Input types
### Required
- **`image`**
    - The input image to be adjusted. This parameter is crucial as it serves as the base for all subsequent color corrections and enhancements.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`strength`**
    - Determines the intensity of the automatic adjustments applied to the image, affecting the overall impact of the correction.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`brightness`**
    - Adjusts the brightness level of the image, influencing its lightness or darkness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`contrast`**
    - Modifies the contrast of the image, enhancing or reducing the difference between its light and dark areas.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`saturation`**
    - Alters the saturation level, adjusting the intensity of the image's colors.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`red`**
    - Adjusts the red color channel, allowing for fine-tuning of the image's red hues.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`green`**
    - Adjusts the green color channel, enabling modifications to the image's green tones.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`blue`**
    - Adjusts the blue color channel, facilitating adjustments to the blue hues in the image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying the auto adjustments, reflecting changes in color properties such as brightness, contrast, saturation, and color channels.
    - Python dtype: `Image`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AutoAdjust:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "strength": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),
                "brightness": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "contrast": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "saturation": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "red": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "green": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "blue": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'auto_adjust'
    CATEGORY = '😺dzNodes/LayerColor'


    def auto_adjust(self, image, strength, brightness, contrast, saturation, red, green, blue):

        if brightness < 0:
            brightness_offset = brightness / 100 + 1
        else:
            brightness_offset = brightness / 50 + 1
        if contrast < 0:
            contrast_offset = contrast / 100 + 1
        else:
            contrast_offset = contrast / 50 + 1
        if saturation < 0:
            saturation_offset = saturation / 100 + 1
        else:
            saturation_offset = saturation / 50 + 1

        red_gamma = self.balance_to_gamma(red)
        green_gamma = self.balance_to_gamma(green)
        blue_gamma = self.balance_to_gamma(blue)


        l_images = []
        l_masks = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))

        max_batch = max(len(l_images), len(l_masks))
        for i in range(max_batch):
            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            orig_image = tensor2pil(_image)
            r, g, b, _  = image_channel_split(orig_image, mode = 'RGB')
            r = normalize_gray(r)
            g = normalize_gray(g)
            b = normalize_gray(b)
            if red:
                r = gamma_trans(r, red_gamma).convert('L')
            if green:
                g = gamma_trans(g, green_gamma).convert('L')
            if blue:
                b = gamma_trans(b, blue_gamma).convert('L')
            ret_image = image_channel_merge((r, g, b), 'RGB')

            if brightness:
                brightness_image = ImageEnhance.Brightness(ret_image)
                ret_image = brightness_image.enhance(factor=brightness_offset)
            if contrast:
                contrast_image = ImageEnhance.Contrast(ret_image)
                ret_image = contrast_image.enhance(factor=contrast_offset)
            if saturation:
                color_image = ImageEnhance.Color(ret_image)
                ret_image = color_image.enhance(factor=saturation_offset)

            ret_image = chop_image_v2(orig_image, ret_image, blend_mode="normal", opacity=strength)
            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

    def balance_to_gamma(self, balance:int) -> float:
        return 0.00005 * balance * balance - 0.01 * balance + 1

```
