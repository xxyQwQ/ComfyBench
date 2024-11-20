---
tags:
- Batch
- Image
- ImageBatch
---

# ImageBatchToList
## Documentation
- Class name: `ImageBatchToList`
- Category: `Bmad/image`
- Output node: `False`

The node is designed to convert a batch of images into a list format, facilitating operations that require individual image processing or manipulation within a batch context. It abstracts the complexity of handling batches of images, making it easier to apply transformations or analyses on each image separately.
## Input types
### Required
- **`images`**
    - Represents the batch of images to be converted into a list. This input is crucial for enabling the node to iterate over each image in the batch and apply necessary transformations or analyses individually.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a list of images that were originally part of a single batch. This allows for individual image processing or manipulation post conversion.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchToList:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "to_list"
    CATEGORY = images_category_path
    OUTPUT_IS_LIST = (True,)

    def to_list(self, images):
        image_list = [images[i][None, ...] for i in range(images.shape[0])]
        return (image_list,)

```
