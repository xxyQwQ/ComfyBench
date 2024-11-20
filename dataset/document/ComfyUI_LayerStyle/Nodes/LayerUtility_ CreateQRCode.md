# LayerUtility: Create QRCode
## Documentation
- Class name: `LayerUtility: CreateQRCode`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

This node generates QR codes based on specified text inputs, allowing for the creation of customizable QR codes with defined sizes and border thicknesses. It encapsulates the process of QR code generation, offering a straightforward way to integrate QR code creation into workflows or applications.
## Input types
### Required
- **`size`**
    - Specifies the size of the generated QR code image. A larger size means the QR code will be bigger, affecting its visibility and scannability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`border`**
    - Determines the thickness of the border around the QR code. A thicker border can help distinguish the QR code from its surroundings, potentially improving its readability.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text`**
    - The text to be encoded into the QR code. This can be any string, such as a URL or a message, which will be converted into a QR code format.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The generated QR code as an image. This output can be used directly in UIs or further processed as needed.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class CreateQRCode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "size": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "border": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = 'create_qrcode'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def create_qrcode(self, size, border, text):
        import qrcode
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=20,
            border=border,
        )
        qr.add_data(text.encode('utf-8'))
        qr.make(fit=True)
        ret_image = qr.make_image(fill_color="black", back_color="white")
        ret_image = ret_image.resize((size, size), Image.BICUBIC)

        return (pil2tensor(ret_image.convert("RGB")), )

```
