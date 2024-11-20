---
tags:
- Batch
- Image
- ImageBatch
---

# Reverse Image Batch
## Documentation
- Class name: `ReverseImageBatch`
- Category: `KJNodes/image`
- Output node: `False`

The ReverseImageBatch node is designed to manipulate a batch of images by reversing their order. This functionality is useful in scenarios where the sequence of images needs to be processed or viewed in reverse order, providing flexibility in handling image data within a batch.
## Input types
### Required
- **`images`**
    - The 'images' parameter represents the batch of images to be reversed. It is crucial for determining the order in which the images will be processed, directly affecting the output by reversing this order.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images with their order reversed from the input batch, allowing for reversed sequence processing or visualization.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ReverseImageBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "reverseimagebatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Reverses the order of the images in a batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
        },
    } 
    
    def reverseimagebatch(self, images):
        reversed_images = torch.flip(images, [0])
        return (reversed_images, )

```
