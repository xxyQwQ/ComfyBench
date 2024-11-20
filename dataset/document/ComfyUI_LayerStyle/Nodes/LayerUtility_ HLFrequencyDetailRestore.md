# LayerUtility: H/L Frequency Detail Restore
## Documentation
- Class name: `LayerUtility: HLFrequencyDetailRestore`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The HLFrequencyDetailRestore node is designed to enhance image details by restoring high and low frequency components. It allows for the manipulation of image textures and details through frequency-based adjustments, providing a means to fine-tune the visual quality of images in a layer-wise manner.
## Input types
### Required
- **`image`**
    - The primary image to which frequency detail restoration will be applied. It serves as the base layer for the operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`detail_image`**
    - An image containing the details to be restored into the primary image. This image is used to extract high-frequency components.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`keep_high_freq`**
    - Determines the intensity of high-frequency details to be preserved in the detail image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`erase_low_freq`**
    - Specifies the level of low-frequency details to be erased from the primary image before merging with high-frequency details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_blur`**
    - Applies a blur to the mask, if provided, to smooth the edges of the applied details.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask`**
    - An optional mask to specify areas of the image where detail restoration should be applied.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after applying high and low frequency detail restoration.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class HLFrequencyDetailRestore:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),
                "detail_image": ("IMAGE",),
                "keep_high_freq": ("INT", {"default": 64, "min": 0, "max": 1023}),
                "erase_low_freq": ("INT", {"default": 32, "min": 0, "max": 1023}),
                "mask_blur": ("INT", {"default": 16, "min": 0, "max": 1023}),
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'hl_frequency_detail_restore'
    CATEGORY = '😺dzNodes/LayerUtility'

    def hl_frequency_detail_restore(self, image, detail_image, keep_high_freq, erase_low_freq, mask_blur, mask=None):

        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        for b in image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in detail_image:
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
        max_batch = max(len(b_images), len(l_images), len(l_masks))

        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            background_image = tensor2pil(background_image).convert('RGB')
            detail_image = l_images[i] if i < len(l_images) else l_images[-1]
            detail_image = tensor2pil(detail_image).convert('RGB')
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            high_ferq = chop_image_v2(ImageChops.invert(detail_image),
                                      gaussian_blur(detail_image, keep_high_freq),
                                      blend_mode='normal', opacity=50)
            high_ferq = ImageChops.invert(high_ferq)
            if erase_low_freq:
                low_freq = gaussian_blur(background_image, erase_low_freq)
            else:
                low_freq = background_image.copy()
            ret_image = chop_image_v2(low_freq, high_ferq, blend_mode="linear light", opacity=100)
            _mask = ImageChops.invert(_mask)
            if mask_blur > 0:
                _mask = gaussian_blur(_mask, mask_blur)
            ret_image.paste(background_image, _mask)
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
