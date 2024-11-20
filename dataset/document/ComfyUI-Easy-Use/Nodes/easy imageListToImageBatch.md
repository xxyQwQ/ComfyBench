---
tags:
- Multimedia
---

# Image List To Image Batch
## Documentation
- Class name: `easy imageListToImageBatch`
- Category: `EasyUse/Image`
- Output node: `False`

This node is designed to transform a list of images into a single image batch, effectively consolidating multiple images into a unified tensor structure for batch processing. It is particularly useful in scenarios where operations need to be applied to multiple images simultaneously, streamlining the workflow by handling images in a batched manner.
## Input types
### Required
- **`images`**
    - Represents the list of images to be batched together. Each image in the list is processed and combined into a single batch, facilitating operations that require batch-level processing.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a single tensor that combines all input images into a batch. This batched format is suitable for further processing or analysis that benefits from batched data handling.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageListToImageBatch:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
      "images": ("IMAGE",),
    }}

  INPUT_IS_LIST = True

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "doit"

  CATEGORY = "EasyUse/Image"

  def doit(self, images):
    if len(images) <= 1:
      return (images[0],)
    else:
      image1 = images[0]
      for image2 in images[1:]:
        if image1.shape[1:] != image2.shape[1:]:
          image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos",
                                              "center").movedim(1, -1)
        image1 = torch.cat((image1, image2), dim=0)
      return (image1,)

```
