---
tags:
- Batch
- Image
- ImageBatch
---

# Insert Images To Batch Indexed
## Documentation
- Class name: `InsertImagesToBatchIndexed`
- Category: `KJNodes/image`
- Output node: `False`

The InsertImagesToBatchIndexed node is designed for integrating specific images into a predefined batch of images at designated positions. It enables precise control over the composition of an image batch by allowing the insertion of new images at specified indices, thus modifying the original batch without altering its overall structure.
## Input types
### Required
- **`original_images`**
    - Specifies the original batch of images where new images will be inserted. It plays a crucial role in determining the base structure of the image batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`images_to_insert`**
    - Defines the images to be inserted into the original batch. This parameter is essential for customizing the content of the image batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`indexes`**
    - Indicates the specific positions within the original image batch where the new images should be inserted. This parameter allows for targeted modifications of the batch.
    - Comfy dtype: `STRING`
    - Python dtype: `list[int]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The modified image batch with the new images inserted at the specified indices.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InsertImagesToBatchIndexed:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "insertimagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Inserts images at the specified indices into the original image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_images": ("IMAGE",),
                "images_to_insert": ("IMAGE",),
                "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
            },
        }
    
    def insertimagesfrombatch(self, original_images, images_to_insert, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Ensure the images_to_insert is a tensor
        if not isinstance(images_to_insert, torch.Tensor):
            images_to_insert = torch.tensor(images_to_insert)
        
        # Insert the images at the specified indices
        for index, image in zip(indices_tensor, images_to_insert):
            original_images[index] = image
        
        return (original_images,)

```
