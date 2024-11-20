---
tags:
- Counting
---

# Get Image Size & Count
## Documentation
- Class name: `GetImageSizeAndCount`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to analyze an image to determine its width, height, and batch size, and then pass the image through unchanged. It serves the purpose of extracting dimensional and quantity information from a batch of images, which can be crucial for further processing steps that require knowledge of the image's size or the total number of images in a batch.
## Input types
### Required
- **`image`**
    - The input image or batch of images for which the size and count information is to be determined. This parameter is essential for the node to perform its analysis and return the dimensions and batch size of the image(s).
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The original image or batch of images passed through the node unchanged.
    - Python dtype: `torch.Tensor`
- **`width`**
    - Comfy dtype: `INT`
    - The width of the input image(s).
    - Python dtype: `int`
- **`height`**
    - Comfy dtype: `INT`
    - The height of the input image(s).
    - Python dtype: `int`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of images in the batch.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImageSizeAndCount:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
        }}

    RETURN_TYPES = ("IMAGE","INT", "INT", "INT",)
    RETURN_NAMES = ("image", "width", "height", "count",)
    FUNCTION = "getsize"
    CATEGORY = "KJNodes/masking"
    DESCRIPTION = """
Returns width, height and batch size of the image,  
and passes it through unchanged.  

"""

    def getsize(self, image):
        width = image.shape[2]
        height = image.shape[1]
        count = image.shape[0]
        return {"ui": {
            "text": [f"{count}x{width}x{height}"]}, 
            "result": (image, width, height, count) 
        }

```
