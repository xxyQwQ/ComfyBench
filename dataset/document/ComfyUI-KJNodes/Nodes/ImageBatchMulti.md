---
tags:
- Batch
- Image
- ImageBatch
---

# Image Batch Multi
## Documentation
- Class name: `ImageBatchMulti`
- Category: `KJNodes/image`
- Output node: `False`

The ImageBatchMulti node is designed to create a batch of images from multiple individual images. It allows for dynamic input count adjustment, enabling the creation of a batch from a varying number of images based on the user's requirements.
## Input types
### Required
- **`inputcount`**
    - Specifies the number of images to be included in the batch. This parameter allows for dynamic adjustment of the batch size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`image_i`**
    - Represents an individual image to be included in the batch. The index i varies based on the input count, starting from 1 (e.g., image_1, image_2, etc.).
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting batch of images, combined from the individual inputs.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchMulti:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 1000, "step": 1}),
                "image_1": ("IMAGE", ),
                "image_2": ("IMAGE", ),
            },
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates an image batch from multiple images.  
You can set how many inputs the node has,  
with the **inputcount** and clicking update.
"""

    def combine(self, inputcount, **kwargs):
        from nodes import ImageBatch
        image_batch_node = ImageBatch()
        image = kwargs["image_1"]
        for c in range(1, inputcount):
            new_image = kwargs[f"image_{c + 1}"]
            image, = image_batch_node.batch(image, new_image)
        return (image,)

```
