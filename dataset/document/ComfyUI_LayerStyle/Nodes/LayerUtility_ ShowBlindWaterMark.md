# LayerUtility: Show BlindWaterMark
## Documentation
- Class name: `LayerUtility: ShowBlindWaterMark`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

This node is designed to decode and reveal blind watermarks embedded within images. It processes each image to extract the watermark, demonstrating the capability to reverse the watermarking process and retrieve hidden information.
## Input types
### Required
- **`image`**
    - The input image from which the blind watermark is to be decoded. This image is expected to have a watermark embedded in it, which this node aims to extract and reveal.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
## Output types
- **`watermark_image`**
    - Comfy dtype: `IMAGE`
    - The output image that contains the decoded watermark, extracted from the input image. This image represents the hidden information that was embedded within the original image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DecodeBlindWaterMark:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),  #
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("watermark_image",)
    FUNCTION = 'watermark_decode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def watermark_decode(self, image):

        NODE_NAME = 'Decode BlindWaterMark'

        ret_images = []

        for i in image:
            _image = torch.unsqueeze(i,0)
            _image = tensor2pil(_image)
            wm_size = watermark_image_size(_image)
            y, u, v, _ = image_channel_split(_image, mode='YCbCr')
            ret_image = decode_watermark(u, wm_size)
            ret_image = ret_image.resize((512, 512), Image.LANCZOS)
            ret_image = normalize_gray(ret_image)
            ret_images.append(pil2tensor(ret_image.convert('RGB')))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), )

```
