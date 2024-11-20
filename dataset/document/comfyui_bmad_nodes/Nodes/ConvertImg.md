---
tags:
- Image
---

# ConvertImg
## Documentation
- Class name: `ConvertImg`
- Category: `Bmad/CV`
- Output node: `False`

The ConvertImg node is designed for explicit image format conversion within a custom node environment, facilitating direct transformations between different image color spaces without resorting to workarounds.
## Input types
### Required
- **`image`**
    - The 'image' parameter represents the input image to be converted. Its role is crucial as it serves as the source image for the conversion process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`to`**
    - The 'to' parameter specifies the target color space format for the conversion, influencing the output image's color representation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is an image that has been converted to the specified color space format, reflecting the changes in color representation as per the 'to' parameter.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ConvertImg:
    """ An explicit conversion, instead of using workarounds when using certain custom nodes. """
    options_map = {
        "RGBA": 4,
        "RGB": 3,
        "GRAY": 1,
    }
    options = list(options_map.keys())

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "to": (cls.options, {"default": cls.options[1]})
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = f"{cv_category_path}"

    def convert(self, image, to):
        image = tensor2opencv(image, self.options_map[to])
        return (opencv2tensor(image),)

```
