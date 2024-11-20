---
tags:
- Latent
- Normalization
---

# Image Normalize -1 to 1
## Documentation
- Class name: `ImageNormalize_Neg1_To_1`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to adjust the pixel values of images so that they fall within the range of -1 to 1, effectively normalizing them for further processing or analysis.
## Input types
### Required
- **`images`**
    - The images to be normalized. This input is crucial for the operation as it directly affects the normalization process by scaling the pixel values.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The normalized images, with pixel values adjusted to fall within the range of -1 to 1.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageNormalize_Neg1_To_1:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                              "images": ("IMAGE",),
    
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "normalize"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Normalize the images to be in the range [-1, 1]  
"""

    def normalize(self,images):
        images = images * 2.0 - 1.0
        return (images,)

```
