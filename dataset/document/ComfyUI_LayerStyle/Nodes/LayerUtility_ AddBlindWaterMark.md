# LayerUtility: Add BlindWaterMark
## Documentation
- Class name: `LayerUtility: AddBlindWaterMark`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

This node is designed to embed a blind watermark into an image, utilizing a method that allows the watermark to be later extracted or detected without altering the visible characteristics of the original image significantly. It operates by manipulating the image's channels to encode the watermark invisibly.
## Input types
### Required
- **`image`**
    - The primary image(s) into which the watermark is to be embedded. This parameter is crucial for determining the base content that will carry the hidden watermark.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`watermark_image`**
    - The image(s) to be used as the watermark. This parameter is essential for defining the content of the watermark that will be invisibly embedded into the primary image(s).
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image(s) after the watermark has been embedded. This output is significant as it represents the original image(s) altered to include the invisible watermark, ready for use or further processing.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class EncodeBlindWaterMark:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE", ),  #
                "watermark_image": ("IMAGE",),  #
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'watermark_encode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def watermark_encode(self, image, watermark_image):

        NODE_NAME = 'Add BlindWaterMark'

        l_images = []
        w_images = []
        ret_images = []

        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
        for w in watermark_image:
            w_images.append(torch.unsqueeze(w, 0))

        for i in range(len(l_images)):
            _image = tensor2pil(l_images[i])
            wm_size = watermark_image_size(_image)
            _wm_image = w_images[i] if i < len(w_images) else w_images[-1]
            _wm_image = tensor2pil(_wm_image)
            _wm_image = _wm_image.resize((wm_size, wm_size), Image.LANCZOS)
            _wm_image = _wm_image.convert("L")

            y, u, v, _ = image_channel_split(_image, mode='YCbCr')
            _u = add_invisibal_watermark(u, _wm_image)
            ret_image = image_channel_merge((y, _u, v), mode='YCbCr')

            if _image.mode == "RGBA":
                ret_image = RGB2RGBA(ret_image, _image.split()[-1])
            ret_images.append(pil2tensor(ret_image))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0),)

```
