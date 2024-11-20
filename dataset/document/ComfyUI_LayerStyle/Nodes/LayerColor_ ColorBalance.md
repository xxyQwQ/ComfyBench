---
tags:
- Color
---

# LayerColor: ColorBalance
## Documentation
- Class name: `LayerColor: ColorBalance`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The ColorBalance node is designed to adjust the color balance of images by modifying the cyan-red, magenta-green, and yellow-blue channels. It allows for fine-tuning of color hues to achieve a desired visual effect, enhancing or correcting the color composition of images.
## Input types
### Required
- **`image`**
    - The input image or images to be processed for color balance adjustment. This parameter is crucial for defining the visual content that will undergo color correction.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`cyan_red`**
    - Adjusts the balance between cyan and red hues in the image, affecting the overall color temperature.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`magenta_green`**
    - Adjusts the balance between magenta and green hues, influencing the color harmony and contrast.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`yellow_blue`**
    - Modifies the balance between yellow and blue hues, impacting the image's warmth or coolness.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image or images after applying the color balance adjustments, reflecting the changes in color hues.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ColorBalance:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "cyan_red": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "magenta_green": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "yellow_blue": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_balance'
    CATEGORY = '😺dzNodes/LayerColor'

    def color_balance(self, image, cyan_red, magenta_green, yellow_blue):

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


        for i in range(len(l_images)):
            _image = l_images[i]
            _mask = l_masks[i]
            orig_image = tensor2pil(_image)

            ret_image = color_balance(orig_image,
                              [cyan_red, magenta_green, yellow_blue],
                              [cyan_red, magenta_green, yellow_blue],
                              [cyan_red, magenta_green, yellow_blue],
                                      shadow_center=0.15,
                                      midtone_center=0.5,
                                      midtone_max=1,
                                      preserve_luminosity=True)

            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
