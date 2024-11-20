---
tags:
- Batch
- Image
- ImageBatch
---

# Replace Images In Batch
## Documentation
- Class name: `ReplaceImagesInBatch`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to replace a subset of images within a batch with a new set of images, starting from a specified index. It facilitates the manipulation of image batches by allowing selective updating of images, which can be useful in various image processing and augmentation tasks.
## Input types
### Required
- **`original_images`**
    - The batch of original images that will be partially replaced. It plays a crucial role in the operation by serving as the base for the replacement process.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`replacement_images`**
    - The images that will replace a portion of the original image batch. Their inclusion directly alters the content of the batch starting from the specified index.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start_index`**
    - The index within the original image batch from which the replacement should begin. It determines the starting point for the replacement operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The modified image batch after the replacement operation has been performed.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ReplaceImagesInBatch:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "replace"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Replaces the images in a batch, starting from the specified start index,  
with the replacement images.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "original_images": ("IMAGE",),
                 "replacement_images": ("IMAGE",),
                 "start_index": ("INT", {"default": 1,"min": 0, "max": 4096, "step": 1}),
        },
    } 
    
    def replace(self, original_images, replacement_images, start_index):
        images = None
        if start_index >= len(original_images):
            raise ValueError("GetImageRangeFromBatch: Start index is out of range")
        end_index = start_index + len(replacement_images)
        if end_index > len(original_images):
            raise ValueError("GetImageRangeFromBatch: End index is out of range")
         # Create a copy of the original_images tensor
        original_images_copy = original_images.clone()
        original_images_copy[start_index:end_index] = replacement_images
        images = original_images_copy
        return (images, )

```
