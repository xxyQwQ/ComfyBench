---
tags:
- Crop
- Image
- ImageTransformation
---

# 🔧 Image Crop
## Documentation
- Class name: `ImageCrop+`
- Category: `essentials/image manipulation`
- Output node: `False`

The ImageCrop+ node is designed for cropping images to a specified width and height starting from a given x and y coordinate. It allows for precise control over the portion of the image to be extracted, making it a fundamental tool for image manipulation and preprocessing tasks.
## Input types
### Required
- **`image`**
    - The input image to be cropped. This parameter is crucial as it defines the source image from which a specific region will be extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the width of the cropped image. It determines how wide the resulting image will be after cropping.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the height of the cropped image. It determines the vertical size of the resulting image after cropping.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`position`**
    - Defines the starting position for the crop operation, offering predefined options for alignment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`x_offset`**
    - The horizontal offset from the specified position for additional control over the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`y_offset`**
    - The vertical offset from the specified position for additional control over the cropping area.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The cropped portion of the input image. This output is the direct result of the cropping operation, representing the specified region of the original image.
    - Python dtype: `torch.Tensor`
- **`x`**
    - Comfy dtype: `INT`
    - The x-coordinate of the top-left corner of the cropped image.
    - Python dtype: `int`
- **`y`**
    - Comfy dtype: `INT`
    - The y-coordinate of the top-left corner of the cropped image.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ImageSharpen](../../Comfy/Nodes/ImageSharpen.md)
    - SetNode
    - IPAdapterApply
    - Reroute
    - [DWPreprocessor](../../comfyui_controlnet_aux/Nodes/DWPreprocessor.md)
    - [LineArtPreprocessor](../../comfyui_controlnet_aux/Nodes/LineArtPreprocessor.md)
    - [MiDaS-DepthMapPreprocessor](../../comfyui_controlnet_aux/Nodes/MiDaS-DepthMapPreprocessor.md)
    - IPAdapterApplyFaceID



## Source code
```python
class ImageCrop:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 256, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "position": (["top-left", "top-center", "top-right", "right-center", "bottom-right", "bottom-center", "bottom-left", "left-center", "center"],),
                "x_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
                "y_offset": ("INT", { "default": 0, "min": -99999, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE","INT","INT",)
    RETURN_NAMES = ("IMAGE","x","y",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, position, x_offset, y_offset):
        _, oh, ow, _ = image.shape

        width = min(ow, width)
        height = min(oh, height)

        if "center" in position:
            x = round((ow-width) / 2)
            y = round((oh-height) / 2)
        if "top" in position:
            y = 0
        if "bottom" in position:
            y = oh-height
        if "left" in position:
            x = 0
        if "right" in position:
            x = ow-width

        x += x_offset
        y += y_offset

        x2 = x+width
        y2 = y+height

        if x2 > ow:
            x2 = ow
        if x < 0:
            x = 0
        if y2 > oh:
            y2 = oh
        if y < 0:
            y = 0

        image = image[:, y:y2, x:x2, :]

        return(image, x, y, )

```
