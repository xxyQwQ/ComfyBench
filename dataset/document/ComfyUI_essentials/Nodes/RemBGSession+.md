---
tags:
- BackgroundRemoval
- Image
---

# 🔧 RemBG Session
## Documentation
- Class name: `RemBGSession+`
- Category: `essentials/image manipulation`
- Output node: `False`

The RemBGSession node is designed for background removal from images, utilizing various models to cater to different purposes such as general image segmentation, human segmentation, and cloth parsing. It supports execution on multiple hardware providers, allowing for flexible deployment.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for background removal, offering a range of options for different segmentation tasks.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`providers`**
    - Determines the hardware provider for executing the model, supporting a variety of platforms including CPU, CUDA, and more.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`rembg_session`**
    - Comfy dtype: `REMBG_SESSION`
    - Produces a session configured for background removal, ready to process images according to the specified model and provider.
    - Python dtype: `rembg.Session`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemBGSession:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["u2net: general purpose", "u2netp: lightweight general purpose", "u2net_human_seg: human segmentation", "u2net_cloth_seg: cloths Parsing", "silueta: very small u2net", "isnet-general-use: general purpose", "isnet-anime: anime illustrations", "sam: general purpose"],),
                "providers": (['CPU', 'CUDA', 'ROCM', 'DirectML', 'OpenVINO', 'CoreML', 'Tensorrt', 'Azure'],),
            },
        }

    RETURN_TYPES = ("REMBG_SESSION",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, model, providers):
        from rembg import new_session

        model = model.split(":")[0]
        return (new_session(model, providers=[providers+"ExecutionProvider"]),)

```
