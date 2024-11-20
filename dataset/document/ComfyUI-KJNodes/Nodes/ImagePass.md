# ImagePass
## Documentation
- Class name: `ImagePass`
- Category: `KJNodes/image`
- Output node: `False`

The ImagePass node is designed to facilitate the unaltered passage of an image through a processing pipeline. It serves as a neutral conduit, ensuring that the input image is passed through without any modifications, alterations, or enhancements.
## Input types
### Required
- **`image`**
    - The 'image' parameter is the primary input for the ImagePass node, representing the image to be passed through unmodified. It plays a crucial role in the node's operation by being the sole piece of data the node acts upon, ensuring the image's integrity is maintained throughout the process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output 'image' parameter is identical to the input image, signifying that the image has been passed through the node without any modifications. It underscores the node's purpose of providing a direct, unaltered conduit for image data.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImagePass:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "passthrough"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Passes the image through without modifying it.
"""

    def passthrough(self, image):
        return image,

```
