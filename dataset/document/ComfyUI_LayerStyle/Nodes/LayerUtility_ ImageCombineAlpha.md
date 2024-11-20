# LayerUtility: ImageCombineAlpha
## Documentation
- Class name: `LayerUtility: ImageCombineAlpha`
- Category: `😺dzNodes/LayerUtility`
- Output node: `False`

The ImageCombineAlpha node is designed to combine RGB images with masks to produce images with an alpha channel, effectively integrating transparency information from the mask into the RGB image. This process allows for the creation of RGBA images where the transparency level is controlled by the mask, enabling more complex image compositing tasks.
## Input types
### Required
- **`RGB_image`**
    - The RGB_image parameter represents the input RGB images that are to be combined with masks to generate RGBA images. This parameter is crucial for defining the visual content of the output images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`mask`**
    - The mask parameter represents the input masks that define the transparency levels for the corresponding RGB images. These masks are essential for integrating transparency into the RGB images to create RGBA outputs.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
### Optional
## Output types
- **`RGBA_image`**
    - Comfy dtype: `IMAGE`
    - The RGBA_image output represents the resulting images with an alpha channel added, combining the input RGB images with the transparency information from the masks.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageCombineAlpha:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        channel_mode = ['RGBA', 'YCbCr', 'LAB', 'HSV']
        return {
            "required": {
                "RGB_image": ("IMAGE", ),  #
                "mask": ("MASK",),  #
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("RGBA_image",)
    FUNCTION = 'image_combine_alpha'
    CATEGORY = '😺dzNodes/LayerUtility'

    def image_combine_alpha(self, RGB_image, mask):

        ret_images = []
        input_images = []
        input_masks = []

        for i in RGB_image:
            input_images.append(torch.unsqueeze(i, 0))
        if mask.dim() == 2:
            mask = torch.unsqueeze(mask, 0)
        for m in mask:
            input_masks.append(torch.unsqueeze(m, 0))

        max_batch = max(len(input_images), len(input_masks))
        for i in range(max_batch):
            _image = input_images[i] if i < len(input_images) else input_images[-1]
            _mask = input_masks[i] if i < len(input_masks) else input_masks[-1]
            r, g, b, _ = image_channel_split(tensor2pil(_image).convert('RGB'), 'RGB')
            ret_image = image_channel_merge((r, g, b, tensor2pil(_mask).convert('L')), 'RGBA')

            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
