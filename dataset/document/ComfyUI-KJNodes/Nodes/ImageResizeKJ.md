---
tags:
- ImageResize
- ImageScaling
- ImageSize
- ImageTransformation
---

# Resize Image
## Documentation
- Class name: `ImageResizeKJ`
- Category: `KJNodes/image`
- Output node: `False`

The ImageResizeKJ node is designed to resize images to specified dimensions, with options to maintain the original aspect ratio, upscale using various methods, and ensure the final dimensions are divisible by a given number. It provides flexibility in defining the target size either directly or based on the size of another image, catering to diverse image processing needs.
## Input types
### Required
- **`image`**
    - The input image to be resized. This parameter is crucial as it provides the source image for the resizing operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`width`**
    - The target width for the resized image. If keep_proportion is true and width is set, the height will be adjusted to maintain the aspect ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - The target height for the resized image. Similar to width, if keep_proportion is true and height is set, the width will be adjusted accordingly.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`upscale_method`**
    - Specifies the method to be used for upscaling the image, offering various algorithms to choose from.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`keep_proportion`**
    - A boolean flag indicating whether to maintain the original aspect ratio of the image during resizing.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`divisible_by`**
    - Ensures the dimensions of the resized image are divisible by this value, useful for certain types of neural network inputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`width_input`**
    - An optional width input that overrides the width parameter if provided, allowing for dynamic resizing based on external inputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height_input`**
    - An optional height input that overrides the height parameter if provided, similar to width_input.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`get_image_size`**
    - An optional image input used to determine the target size for resizing, based on the dimensions of this image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - unknown
    - Python dtype: `unknown`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the resized image.
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the resized image.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageResizeKJ:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "upscale_method": (s.upscale_methods,),
                "keep_proportion": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "width_input": ("INT", { "forceInput": True}),
                "height_input": ("INT", { "forceInput": True}),
                "get_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Resizes the image to the specified width and height.  
Size can be retrieved from the inputs, and the final scale  
is  determined in this order of importance:  
- get_image_size  
- width_input and height_input  
- width and height widgets  
  
Keep proportions keeps the aspect ratio of the image, by  
highest dimension.  
"""

    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, width_input=None, height_input=None, get_image_size=None):
        B, H, W, C = image.shape
        if width_input:
            width = width_input
        if height_input:
            height = height_input
        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape

        if keep_proportion and get_image_size is None:
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            if width == 0 and height != 0:
                ratio = height / H
                width = round(W * ratio)
            elif height == 0 and width != 0:
                ratio = width / W
                height = round(H * ratio)
            elif width != 0 and height != 0:
                # Scale based on which dimension is smaller in proportion to the desired dimensions
                ratio = min(width / W, height / H)
                width = round(W * ratio)
                height = round(H * ratio)
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H
    
        if divisible_by > 1 and get_image_size is None:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        image = image.movedim(-1,1)
        scaled = common_upscale(image, width, height, upscale_method, 'disabled')
        scaled = scaled.movedim(1,-1)

        return(scaled, scaled.shape[2], scaled.shape[1],)

```
