---
tags:
- Batch
- Image
- ImageBatch
---

# 🔧 Image From Batch
## Documentation
- Class name: `ImageFromBatch+`
- Category: `essentials/image batch`
- Output node: `False`

This node is designed to extract a specific segment from a batch of images based on the starting index and the length of the segment desired. It enables selective retrieval of images from a larger batch, facilitating operations that require processing or analysis of subsets of image collections.
## Input types
### Required
- **`image`**
    - The batch of images from which a segment is to be extracted. It serves as the primary input for determining the subset of images to be processed.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`start`**
    - The starting index from which the image segment extraction begins. It defines the initial position in the batch from which images are to be retrieved, allowing for targeted selection within the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`length`**
    - Specifies the number of images to extract from the batch, starting from the 'start' index. It determines the size of the image segment to be retrieved, enabling precise control over the subset size.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The extracted segment of images from the specified batch. This output facilitates further processing or analysis of the selected subset of images.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "start": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "length": ("INT", { "default": -1, "min": -1, "step": 1, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image, start, length):
        if length<0:
            length = image.shape[0]
        start = min(start, image.shape[0]-1)
        length = min(image.shape[0]-start, length)
        return (image[start:start + length], )

```
