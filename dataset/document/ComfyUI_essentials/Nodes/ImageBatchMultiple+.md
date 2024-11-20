---
tags:
- Batch
- Image
- ImageBatch
---

# 🔧 Images Batch Multiple
## Documentation
- Class name: `ImageBatchMultiple+`
- Category: `essentials/image batch`
- Output node: `False`

The node 'ImageBatchMultiple+' is designed to handle multiple images simultaneously, applying operations that involve combining, transforming, or otherwise processing these images in a batch-oriented manner. It abstracts the complexity of dealing with image dimensions and formats, providing a streamlined approach to batch image manipulation.
## Input types
### Required
- **`image_i`**
    - Specifies the input images to be processed. It is central to the node's operation, as it determines the images that will undergo manipulation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`method`**
    - Specifies the approach for resizing or manipulating the images, such as stretching or cropping, which affects the final appearance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The manipulated images after processing according to the specified parameters.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageBatchMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image_1, method, image_2=None, image_3=None, image_4=None, image_5=None):
        out = image_1

        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((image_1, image_2), dim=0)
        if image_3 is not None:
            if image_1.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_3), dim=0)
        if image_4 is not None:
            if image_1.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_4), dim=0)
        if image_5 is not None:
            if image_1.shape[1:] != image_5.shape[1:]:
                image_5 = comfy.utils.common_upscale(image_5.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_5), dim=0)

        return (out,)

```
