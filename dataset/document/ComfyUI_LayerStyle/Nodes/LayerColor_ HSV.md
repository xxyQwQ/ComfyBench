---
tags:
- Color
---

# LayerColor: HSV
## Documentation
- Class name: `LayerColor: HSV`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

This node is designed for color correction in the HSV color space, allowing adjustments to the hue, saturation, and value (brightness) of an image. It enables fine-tuning of image colors to achieve desired visual effects or correct color imbalances.
## Input types
### Required
- **`image`**
    - The input image to be color corrected. This parameter is crucial as it determines the base for all subsequent HSV adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`H`**
    - Specifies the hue adjustment. Positive values shift the hue right, while negative values shift it left, enabling the modification of the image's overall color tone.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`S`**
    - Determines the saturation level adjustment. Positive values increase saturation, making colors more vivid, while negative values decrease it, leading to a more muted color palette.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`V`**
    - Controls the value (brightness) adjustment. Positive values make the image brighter, whereas negative values make it darker, affecting the image's lightness or darkness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after applying HSV color corrections. This reflects the adjustments made to hue, saturation, and value, showcasing the color-corrected version of the input image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorCorrectHSV:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "H": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "S": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
                "V": ("INT", {"default": 0, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_correct_HSV'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_correct_HSV(self, image, H, S, V):

        ret_images = []

        for i in image:
            i = torch.unsqueeze(i,0)
            __image = tensor2pil(i)
            _h, _s, _v = tensor2pil(i).convert('HSV').split()
            if H != 0 :
                _h = image_hue_offset(_h, H)
            if S != 0 :
                _s = image_gray_offset(_s, S)
            if V != 0 :
                _v = image_gray_offset(_v, V)
            ret_image = image_channel_merge((_h, _s, _v), 'HSV')

            if __image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, __image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
