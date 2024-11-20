# imageSplitGrid
## Documentation
- Class name: `easy imageSplitGrid`
- Category: `EasyUse/Image`
- Output node: `False`

The `easy imageSplitGrid` node is designed for splitting a single image into multiple smaller images based on specified grid dimensions. It allows for the division of an image into a grid of smaller, equally sized images, facilitating operations that require processing or analyzing parts of an image separately.
## Input types
### Required
- **`images`**
    - The input image to be split into a grid. This image is divided according to the specified row and column parameters, resulting in multiple smaller images.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`row`**
    - Specifies the number of rows in the grid. This determines how many horizontal splits will be made on the input image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`column`**
    - Specifies the number of columns in the grid. This determines how many vertical splits will be made on the input image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`images`**
    - Comfy dtype: `IMAGE`
    - The output is a batch of images that have been split according to the specified grid dimensions. Each image in the batch represents a segment of the original image as defined by the grid.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class imageSplitGrid:
  @classmethod
  def INPUT_TYPES(s):
    return {
      "required": {
        "images": ("IMAGE",),
        "row": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
        "column": ("INT", {"default": 1,"min": 1,"max": 10,"step": 1,}),
      }
    }

  RETURN_TYPES = ("IMAGE",)
  RETURN_NAMES = ("images",)
  FUNCTION = "doit"
  CATEGORY = "EasyUse/Image"

  def crop(self, image, width, height, x, y):
    x = min(x, image.shape[2] - 1)
    y = min(y, image.shape[1] - 1)
    to_x = width + x
    to_y = height + y
    img = image[:, y:to_y, x:to_x, :]
    return img

  def doit(self, images, row, column):
    _, height, width, _ = images.shape
    sub_width = width // column
    sub_height = height // row
    new_images = []
    for i in range(row):
        for j in range(column):
          new_images.append(self.crop(images, sub_width, sub_height, j * sub_width, i * sub_height))

    return (torch.cat(new_images, dim=0),)

```
