---
tags:
- ImageTransformation
---

# Remap Depth
## Documentation
- Class name: `RemapDepth`
- Category: `Marigold`
- Output node: `False`

The RemapDepth node is designed to adjust the depth values of an image within a specified range and optionally clamp these values for normalization. It operates by scaling the depth values linearly between a minimum and maximum value, enhancing the depth perception or preparing the image for further processing steps.
## Input types
### Required
- **`image`**
    - The input image whose depth values are to be remapped. This image is the primary subject of the node's operation, with its depth values being adjusted according to the other parameters.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`min`**
    - The minimum value in the target range for depth remapping. It sets the lower bound for the depth values in the output image, playing a crucial role in defining the depth scale.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max`**
    - The maximum value in the target range for depth remapping. It sets the upper bound for the depth values in the output image, crucial for defining the scale and contrast of depth perception.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clamp`**
    - A boolean parameter that determines whether the remapped depth values should be clamped to the range [0.0, 1.0]. Clamping ensures that all depth values stay within a normalized range, enhancing compatibility with subsequent processing or visualization steps.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output image with depth values remapped and optionally clamped. This image is ready for further processing or visualization, with enhanced depth perception or normalization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RemapDepth:
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

    CATEGORY = "Marigold"
        
    def remap(self, image, min, max, clamp):
        if image.dtype == torch.float16:
            image = image.to(torch.float32)
        image = min + image * (max - min)
        if clamp:
            image = torch.clamp(image, min=0.0, max=1.0)
        return (image, )

```
