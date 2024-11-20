---
tags:
- Color
---

# LayerColor: AutoBrightness
## Documentation
- Class name: `LayerColor: AutoBrightness`
- Category: `😺dzNodes/LayerColor`
- Output node: `False`

The AutoBrightness node automatically adjusts the brightness of an image based on specified strength and saturation levels, optionally considering a mask to apply the adjustment selectively.
## Input types
### Required
- **`image`**
    - The image to be processed for auto brightness adjustment. It serves as the primary input for the node's operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`strength`**
    - Determines the intensity of the brightness adjustment. A higher value results in a more pronounced change.
    - Comfy dtype: `INT`
    - Python dtype: `INT`
- **`saturation`**
    - Adjusts the saturation level of the image alongside brightness, allowing for a balanced color correction.
    - Comfy dtype: `INT`
    - Python dtype: `INT`
### Optional
- **`mask`**
    - An optional mask that can be applied to restrict the brightness adjustment to specific areas of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `MASK`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image after auto brightness adjustment, reflecting changes in brightness and saturation.
    - Python dtype: `IMAGE`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AutoBrightness:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "strength": ("INT", {"default": 75, "min": 0, "max": 100, "step": 1}),
                "saturation": ("INT", {"default": 8, "min": -255, "max": 255, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'auto_brightness'
    CATEGORY = '😺dzNodes/LayerColor'

    def auto_brightness(self, image, strength, saturation, mask=None):

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
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))
        max_batch = max(len(l_images), len(l_masks))
        for i in range(max_batch):
            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            orig_image = tensor2pil(_image)

            _l, _a, _b = orig_image.convert('LAB').split()
            _histogram = histogram_equalization(_l, _mask, gamma_strength=strength/100)
            _l = chop_image(_l, _histogram, 'normal', strength)
            ret_image = image_channel_merge((_l, _a, _b), 'LAB')
            if saturation != 0 :
                _h, _s, _v = ret_image.convert('HSV').split()
                _s = image_gray_offset(_s, saturation)
                ret_image = image_channel_merge((_h, _s, _v), 'HSV')

            if orig_image.mode == 'RGBA':
                ret_image = RGB2RGBA(ret_image, orig_image.split()[-1])

            ret_images.append(pil2tensor(ret_image))
        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
