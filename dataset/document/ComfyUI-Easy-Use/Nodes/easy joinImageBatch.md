---
tags:
- Batch
- Image
- ImageBatch
---

# JoinImageBatch
## Documentation
- Class name: `easy joinImageBatch`
- Category: `EasyUse/Image`
- Output node: `False`

This node is designed to merge a batch of images into a single large image, either by stacking them horizontally or vertically based on the specified mode. It abstracts the complexity of image manipulation and resizing, providing a straightforward way to create composite images from multiple inputs.
## Input types
### Required
- **`images`**
    - A batch of images to be joined into a single image. The importance of this parameter lies in its role as the primary input that determines the content and structure of the output image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mode`**
    - Specifies the orientation for joining images, either 'horizontal' or 'vertical'. This affects the final layout of the composite image, influencing its visual presentation and dimensions.
    - Comfy dtype: `['horizontal', 'vertical']`
    - Python dtype: `Tuple[str, Dict[str, Any]]`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting single large image created by joining the input batch of images according to the specified mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class JoinImageBatch:
  """Turns an image batch into one big image."""

  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "mode": (("horizontal", "vertical"), {"default": "horizontal"}),
      },
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("image",)
  FUNCTION = "join"
  CATEGORY = "EasyUse/Image"

  def join(self, images, mode):
    n, h, w, c = images.shape
    image = None
    if mode == "vertical":
      # for vertical we can just reshape
      image = images.reshape(1, n * h, w, c)
    elif mode == "horizontal":
      # for horizontal we have to swap axes
      image = torch.transpose(torch.transpose(images, 1, 2).reshape(1, n * w, h, c), 1, 2)
    return (image,)

```
