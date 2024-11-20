---
tags:
- Multimedia
---

# Image Batch To Image List
## Documentation
- Class name: `easy imageBatchToImageList`
- Category: `EasyUse/Image`
- Output node: `False`

The node is designed to convert a batch of images into a list of individual images, facilitating operations that require handling images on a per-image basis rather than in batch form.
## Input types
### Required
- **`image`**
    - The input image batch to be split into individual images. This parameter is crucial for the node's operation as it directly influences the output list of images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a list of individual images that were originally part of the input image batch.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageBatchToImageList:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {"image": ("IMAGE",), }}

  RETURN_TYPES = ("IMAGE",)
  OUTPUT_IS_LIST = (True,)
  FUNCTION = "doit"

  CATEGORY = "EasyUse/Image"

  def doit(self, image):
    images = [image[i:i + 1, ...] for i in range(image.shape[0])]
    return (images,)

```
