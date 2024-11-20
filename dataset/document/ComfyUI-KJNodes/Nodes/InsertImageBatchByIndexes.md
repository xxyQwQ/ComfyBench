---
tags:
- Batch
- Image
- ImageBatch
---

# Insert Image Batch By Indexes
## Documentation
- Class name: `InsertImageBatchByIndexes`
- Category: `KJNodes/image`
- Output node: `False`

The InsertImageBatchByIndexes node is designed for integrating a sequence of images into an existing batch of images at specified indices. It ensures that the newly inserted images are placed at the correct positions within the original batch, maintaining the original sequence order where possible. This node is particularly useful in workflows requiring dynamic modification or augmentation of image batches, such as in data preprocessing for machine learning models.
## Input types
### Required
- **`images`**
    - The original batch of images into which new images will be inserted. It serves as the base for the insertion operation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`images_to_insert`**
    - The images to be inserted into the original batch at specified indices. This parameter allows for dynamic content addition to the image batch.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`insert_indexes`**
    - The indices within the original image batch where the new images should be inserted. These indices dictate the exact positions for insertion, aligning with the sequence of 'images_to_insert'.
    - Comfy dtype: `INDEXES`
    - Python dtype: `List[int]`
## Output types
- **`images_after_insert`**
    - Comfy dtype: `IMAGE`
    - The updated batch of images after the insertion operation, maintaining the original sequence order with the new images integrated at specified positions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InsertImageBatchByIndexes:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",), 
                "images_to_insert": ("IMAGE",), 
                "insert_indexes": ("INDEXES",),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images_after_insert", )
    FUNCTION = "insert"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
This node is designed to be use with node FilterZeroMasksAndCorrespondingImages
It inserts the images_to_insert into images according to insert_indexes

Returns:
    images_after_insert: updated original images with origonal sequence order
"""
    
    def insert(self, images, images_to_insert, insert_indexes):        
        images_after_insert = images
        
        if images_to_insert is not None and insert_indexes is not None:
            images_to_insert_num = len(images_to_insert)
            insert_indexes_num = len(insert_indexes)
            if images_to_insert_num == insert_indexes_num:
                images_after_insert = []

                i_images = 0
                for i in range(len(images) + images_to_insert_num):
                    if i in insert_indexes:
                        images_after_insert.append(images_to_insert[insert_indexes.index(i)])
                    else:
                        images_after_insert.append(images[i_images])
                        i_images += 1
                        
                images_after_insert = torch.stack(images_after_insert, dim=0)
                
            else:
                print(f"[WARNING] skip this node, due to number of images_to_insert ({images_to_insert_num}) is not equal to number of insert_indexes ({insert_indexes_num})")


        return (images_after_insert, )

```
