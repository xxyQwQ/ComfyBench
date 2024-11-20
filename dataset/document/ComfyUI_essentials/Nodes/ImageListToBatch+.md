---
tags:
- Batch
- Image
- ImageBatch
---

# 🔧 Image List To Batch
## Documentation
- Class name: `ImageListToBatch+`
- Category: `essentials/image batch`
- Output node: `False`

This node is designed to convert a list of images into a single batched tensor, ensuring that all images conform to the same dimensions through optional resizing and cropping. It's particularly useful for preparing a collection of images for batch processing in machine learning models, where uniform input sizes are often required.
## Input types
### Required
- **`image`**
    - The list of images to be batched together. Each image in the list is processed to match the dimensions of the first image, using bicubic upsampling and center cropping if necessary.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A single tensor containing all input images batched together. This tensor is suitable for direct use in models that require batched input.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageListToBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    INPUT_IS_LIST = True
    CATEGORY = "essentials/image batch"

    def execute(self, image):
        shape = image[0].shape[1:3]
        out = []

        for i in range(len(image)):
            img = image[i]
            if image[i].shape[1:3] != shape:
                img = comfy.utils.common_upscale(img.permute([0,3,1,2]), shape[1], shape[0], upscale_method='bicubic', crop='center').permute([0,2,3,1])
            out.append(img)

        out = torch.cat(out, dim=0)

        return (out,)

```
