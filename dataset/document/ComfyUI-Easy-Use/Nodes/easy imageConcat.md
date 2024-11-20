---
tags:
- Image
---

# imageConcat
## Documentation
- Class name: `easy imageConcat`
- Category: `EasyUse/Image`
- Output node: `False`

The `easy imageConcat` node is designed to concatenate two images either horizontally or vertically, with an option to match their sizes for seamless integration. This functionality is essential for tasks requiring the combination of images in a specific orientation, such as creating panoramic views or composite images.
## Input types
### Required
- **`image1`**
    - The first image to be concatenated. It serves as the base image to which the second image will be attached.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`image2`**
    - The second image to be concatenated to the first one. This image can be resized to match the first image's dimensions if required.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`direction`**
    - Specifies the direction of concatenation ('right', 'down', 'left', 'up'), determining how the second image is attached to the first.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`match_image_size`**
    - A boolean flag indicating whether the second image should be resized to match the first image's dimensions before concatenation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The resulting image after concatenating the two input images according to the specified direction and size matching option.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class imageConcat:
  @classmethod
  def INPUT_TYPES(s):
    return {"required": {
      "image1": ("IMAGE",),
      "image2": ("IMAGE",),
      "direction": (['right','down','left','up',],{"default": 'right'}),
      "match_image_size": ("BOOLEAN", {"default": False}),
    }}

  RETURN_TYPES = ("IMAGE",)
  FUNCTION = "concat"
  CATEGORY = "EasyUse/Image"

  def concat(self, image1, image2, direction, match_image_size):
    if match_image_size:
      image2 = torch.nn.functional.interpolate(image2, size=(image1.shape[2], image1.shape[3]), mode="bilinear")
    if direction == 'right':
      row = torch.cat((image1, image2), dim=2)
    elif direction == 'down':
      row = torch.cat((image1, image2), dim=1)
    elif direction == 'left':
      row = torch.cat((image2, image1), dim=2)
    elif direction == 'up':
      row = torch.cat((image2, image1), dim=1)
    return (row,)

```
