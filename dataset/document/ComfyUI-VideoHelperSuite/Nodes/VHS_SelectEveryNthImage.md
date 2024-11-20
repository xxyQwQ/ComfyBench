---
tags:
- Image
- Multimedia
---

# Select Every Nth Image 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthImage`
- Category: `Video Helper Suite 🎥🅥🅗🅢/image`
- Output node: `False`

The node 'VHS_SelectEveryNthImage' is designed to filter a batch of images by selecting every nth image from the batch, optionally skipping a specified number of initial images. This functionality is useful for thinning out image datasets or sequences to reduce processing load or to select a subset of images for analysis or display.
## Input types
### Required
- **`images`**
    - The 'images' parameter represents the batch of images to be filtered. It is crucial for determining which images are selected for output based on the specified selection criteria.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`select_every_nth`**
    - This parameter specifies the interval at which images are selected from the batch. It plays a key role in determining the density of the output image subset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_images`**
    - Defines the number of initial images to skip before starting the selection process. This allows for greater control over which images are included in the output subset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The output consists of a subset of images selected according to the specified criteria.
    - Python dtype: `torch.Tensor`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of images selected and returned by the node.
    - Python dtype: `int`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "images": ("IMAGE",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_images": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                    
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/image"

    RETURN_TYPES = ("IMAGE", "INT",)
    RETURN_NAMES = ("IMAGE", "count",)
    FUNCTION = "select_images"

    def select_images(self, images: Tensor, select_every_nth: int, skip_first_images: int):
        sub_images = images[skip_first_images::select_every_nth]
        return (sub_images, sub_images.size(0))

```
