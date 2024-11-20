---
tags:
- ImageTransformation
---

# Remap Image Range
## Documentation
- Class name: `RemapImageRange`
- Category: `KJNodes/image`
- Output node: `False`

The RemapImageRange node is designed to adjust the pixel value range of an input image to a specified new range. It supports optional clamping to ensure that the remapped image values stay within a desired interval, enhancing flexibility in image preprocessing for various applications.
## Input types
### Required
- **`image`**
    - The input image to be remapped. This parameter is crucial for defining the source image whose pixel values are to be adjusted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`min`**
    - Specifies the minimum value of the new range for the image pixel values. It plays a key role in defining the lower bound of the target range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max`**
    - Defines the maximum value of the new range for the image pixel values, setting the upper limit of the target range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clamp`**
    - A boolean flag that determines whether the remapped image values should be clamped to the [0.0, 1.0] range, ensuring that pixel values remain valid.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with its pixel values remapped to the specified new range. This parameter signifies the transformed image ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemapImageRange:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "image": ("IMAGE",),
            "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
            "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            "clamp": ("BOOLEAN", {"default": True}),
            },
            }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remap"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Remaps the image values to the specified range. 
"""
        
    def remap(self, image, min, max, clamp):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = min + image * (max - min)
        if clamp:
            image = torch.clamp(image, min=0.0, max=1.0)
        return (image, )

```
