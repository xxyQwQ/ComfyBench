---
tags:
- Image
- ImageTransformation
---

# 🔧 Image To Device
## Documentation
- Class name: `ImageToDevice+`
- Category: `essentials/image utils`
- Output node: `False`

The ImageToDevice node is designed for transferring images to a specified computing device, such as GPU or CPU, within a deep learning environment. It supports dynamic device selection based on user input, enabling optimized computation by leveraging the appropriate hardware resources.
## Input types
### Required
- **`image`**
    - The image to be transferred to the specified device. It plays a crucial role in ensuring that the image is processed on the correct hardware, affecting performance and efficiency.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`device`**
    - Specifies the target device for the image transfer. It can be set to 'auto', 'cpu', or 'gpu', allowing for flexible execution across different hardware configurations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The image after being transferred to the specified device, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageToDevice:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image utils"

    def execute(self, image, device):
        if "gpu" == device:
            device = comfy.model_management.get_torch_device()
        elif "auto" == device:
            device = comfy.model_management.intermediate_device()
        else:
            device = 'cpu'

        image = image.clone().to(device)
        torch.cuda.empty_cache()

        return (image,)

```
