---
tags:
- Batch
- Image
- ImageBatch
---

# Get Image Range From Batch
## Documentation
- Class name: `GetImageRangeFromBatch`
- Category: `KJNodes/image`
- Output node: `False`

This node is designed to extract a specific range of images from a given batch based on a starting index and the number of frames desired. It can optionally handle masks associated with the images, ensuring that both images and their corresponding masks are selected in tandem.
## Input types
### Required
- **`images`**
    - The collection of images from which a range will be selected. This parameter is crucial for defining the subset of images to be extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`start_index`**
    - Specifies the starting index from which images will be selected in the batch. A special value of -1 indicates selection from the end.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`num_frames`**
    - Determines the number of images to select from the starting index, defining the size of the output batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`masks`**
    - An optional collection of masks corresponding to the input images. If provided, masks for the selected image range are also returned.
    - Comfy dtype: `MASK`
    - Python dtype: `Optional[List[torch.Tensor]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The selected range of images from the input batch.
    - Python dtype: `List[torch.Tensor]`
- **`mask`**
    - Comfy dtype: `MASK`
    - The masks corresponding to the selected range of images, if masks were provided.
    - Python dtype: `Optional[List[torch.Tensor]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GetImageRangeFromBatch:
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "KJNodes/image"
    DESCRIPTION = """
Creates a new batch using images from the input,  
batch, starting from start_index.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "images": ("IMAGE",),
                 "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                 "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
        },
        "optional": {
            "masks": ("MASK",),
        }
    } 
    
    def imagesfrombatch(self, images, start_index, num_frames, masks=None):
        if start_index == -1:
            start_index = len(images) - num_frames
        if start_index < 0 or start_index >= len(images):
            raise ValueError("GetImageRangeFromBatch: Start index is out of range")
        end_index = start_index + num_frames
        if end_index > len(images):
            raise ValueError("GetImageRangeFromBatch: End index is out of range")
        chosen_images = images[start_index:end_index]
        chosen_masks = masks[start_index:end_index] if masks is not None else None
        return (chosen_images, chosen_masks,)

```
