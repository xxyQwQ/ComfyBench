---
tags:
- Batch
- Image
- ImageBatch
---

# Get Images From Batch Indexed
## Documentation
- Class name: `GetImagesFromBatchIndexed`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to select and return specific images from a given batch based on the provided indices. It allows for the dynamic extraction of a subset of images from a larger collection, facilitating operations that require targeted manipulation or analysis of image batches.
## Input types
### Required
- **`images`**
    - Represents the batch of images from which specific items will be selected. It is crucial for determining the scope of images available for extraction.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`indexes`**
    - A string of comma-separated indices indicating which images to extract from the batch. This parameter directly influences which images are selected and returned.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a subset of images selected from the input batch based on the specified indices.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetImagesFromBatchIndexed:
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "indexedimagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Selects and returns the images at the specified indices as an image batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
        },
    } 
    
    def indexedimagesfrombatch(self, images, indexes):
        
        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Select the images at the specified indices
        chosen_images = images[indices_tensor]
        
        return (chosen_images,)

```
