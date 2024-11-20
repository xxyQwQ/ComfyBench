# LayerUtility: ImageOpacity
## Documentation
- Class name: `LayerUtility: ImageOpacity`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageOpacity node is designed to adjust the opacity of an image, optionally applying an inversion mask to selectively modify the transparency across the image. It provides a way to dynamically control the visual appearance of images by altering their transparency levels.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image to which the opacity adjustment will be applied. It is essential for defining the base visual content that will undergo transparency modification.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`opacity`**
    - The 'opacity' parameter specifies the degree of transparency to apply to the image, with a range from 0 (completely transparent) to 100 (fully opaque). It allows for fine-tuning the visibility of the image.
    - Comfy dtype: `INT`
    - Python dtype: `INT`
- **`invert_mask`**
    - The 'invert_mask' parameter determines whether to invert the mask applied to the image, enabling selective opacity adjustments across different regions of the image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `BOOLEAN`
### Optional
- **`mask`**
    - The 'mask' parameter, when provided, specifies a mask to apply to the image for targeted opacity adjustments. It is optional and used for more complex transparency effects.
    - Comfy dtype: `MASK`
    - Python dtype: `MASK`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The modified image with adjusted opacity.
    - Python dtype: `IMAGE`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask used for opacity adjustment, if any was applied.
    - Python dtype: `MASK`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImageOpacity:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "opacity": ("INT", {"default": 100, "min": 0, "max": 100, "step": 1}),  # 透明度
                "invert_mask": ("BOOLEAN", {"default": True}),  # 反转mask
            },
            "optional": {
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = 'image_opacity'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_opacity(self, image, opacity, invert_mask,
                      mask=None,
                      ):

        ret_images = []
        ret_masks = []
        l_images = []
        l_masks = []
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))

        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            l_masks = []
            for m in mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(l_images), len(l_masks))

        for i in range(max_batch):
            _image = l_images[i] if i < len(l_images) else l_images[-1]
            _image = tensor2pil(_image)
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]
            if invert_mask:
                _color = Image.new("L", _image.size, color=('white'))
                _mask = ImageChops.invert(_mask)
            else:
                _color = Image.new("L", _image.size, color=('black'))

            alpha = 1 - opacity / 100.0
            ret_mask = Image.blend(_mask, _color, alpha)
            R, G, B, = _image.convert('RGB').split()
            if invert_mask:
                ret_mask = ImageChops.invert(ret_mask)
            ret_image = Image.merge('RGBA', (R, G, B, ret_mask))

            ret_images.append(pil2tensor(ret_image))
            ret_masks.append(image2mask(ret_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

```
