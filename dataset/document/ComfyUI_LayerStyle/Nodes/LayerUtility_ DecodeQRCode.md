# LayerUtility: Decode QRCode
## Documentation
- Class name: `LayerUtility: DecodeQRCode`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

The DecodeQRCode node is designed to interpret QR codes from images, applying optional preprocessing to enhance readability before decoding. It aims to extract textual information encoded within QR codes, offering a means to retrieve embedded data from visual inputs.
## Input types
### Required
- **`image`**
    - The image input represents the visual source from which QR codes are to be decoded. It plays a crucial role in the node's operation by providing the raw data necessary for QR code interpretation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`pre_blur`**
    - The pre_blur parameter allows for optional preprocessing of the image to improve QR code readability by applying a Gaussian blur. This can enhance the node's ability to accurately decode QR codes from images that may be noisy or of low quality.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The output is a string or a list of strings representing the decoded textual information from the QR codes found within the input images.
    - Python dtype: `List[str]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DecodeQRCode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "image": ("IMAGE",),
                "pre_blur": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("string", )
    FUNCTION = 'decode_qrcode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def decode_qrcode(self, image, pre_blur):
        ret_texts = []
        from pyzbar.pyzbar import decode
        for i in image:
            _image = torch.unsqueeze(i, 0)
            _image = tensor2pil(_image)
            if pre_blur:
                _image = gaussian_blur(_image, pre_blur)
            qrmessage = decode(_image)
            if len(qrmessage) > 0:
                ret_texts.append(qrmessage[0][0].decode('utf-8'))
            else:
                ret_texts.append("Cannot recognize QR")

        return (ret_texts, )

```
