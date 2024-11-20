---
tags:
- ImageResize
- ImageScaling
- ImageSize
- ImageTransformation
---

# 🔧 Image Resize
## Documentation
- Class name: `ImageResize+`
- Category: `essentials/image manipulation`
- Output node: `False`

This node is designed for resizing images to a specified width and height, offering a straightforward way to adjust image dimensions for various applications. It abstracts the complexities of image processing, providing an easy-to-use interface for resizing operations.
## Input types
### Required
- **`image`**
    - The input image to be resized. This parameter is crucial as it determines the source image that will undergo the resizing process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - Specifies the desired width of the resized image. This parameter directly influences the output image's dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Specifies the desired height of the resized image. This parameter directly influences the output image's dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolation`**
    - Defines the method used for interpolating pixels in the resizing process, affecting the quality and appearance of the resized image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`method`**
    - Determines how the image is resized, influencing whether and how the aspect ratio is preserved or altered.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`condition`**
    - Specifies under what conditions the resizing should occur, allowing for conditional resizing based on the original and target dimensions.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`multiple_of`**
    - Ensures the dimensions of the resized image are multiples of this value, useful for certain processing or model input requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The resized image. This output reflects the changes in dimensions specified by the width and height parameters.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The actual width of the resized image, which may differ from the requested width due to constraints like maintaining aspect ratio or alignment to a multiple.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The actual height of the resized image, which may differ from the requested height due to constraints like maintaining aspect ratio or alignment to a multiple.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - SetNode
    - [SVD_img2vid_Conditioning](../../Comfy/Nodes/SVD_img2vid_Conditioning.md)
    - [PrepImageForClipVision](../../ComfyUI_IPAdapter_plus/Nodes/PrepImageForClipVision.md)



## Source code
```python
class ImageResize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "interpolation": (["nearest", "bilinear", "bicubic", "area", "nearest-exact", "lanczos"],),
                "method": (["stretch", "keep proportion", "fill / crop", "pad"],),
                "condition": (["always", "downscale if bigger", "upscale if smaller", "if bigger area", "if smaller area"],),
                "multiple_of": ("INT", { "default": 0, "min": 0, "max": 512, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, width, height, method="stretch", interpolation="nearest", condition="always", multiple_of=0, keep_proportion=False):
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion:
            method = "keep proportion"

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if method == 'keep proportion' or method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = ow

            ratio = min(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)

            if method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        elif method.startswith('fill'):
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow*ratio)
            new_height = round(oh*ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        if "always" in condition \
            or ("downscale if bigger" == condition and (oh > height or ow > width)) or ("upscale if smaller" == condition and (oh < height or ow < width)) \
            or ("bigger area" in condition and (oh * ow > height * width)) or ("smaller area" in condition and (oh * ow < height * width)):

            outputs = image.permute(0,3,1,2)

            if interpolation == "lanczos":
                outputs = comfy.utils.lanczos(outputs, width, height)
            else:
                outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

            if method == 'pad':
                if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                    outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

            outputs = outputs.permute(0,2,3,1)

            if method.startswith('fill'):
                if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                    outputs = outputs[:, y:y2, x:x2, :]
        else:
            outputs = image

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)

        return(outputs, outputs.shape[2], outputs.shape[1],)

```
