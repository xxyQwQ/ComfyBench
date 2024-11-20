---
tags:
- GridLayout
- Image
- Tiled
---

# Repeat Into Grid (image)
## Documentation
- Class name: `Repeat Into Grid (image)`
- Category: `Bmad/image`
- Output node: `False`

This node tiles input image samples into a grid of configurable dimensions, effectively repeating the input image across a specified number of rows and columns to create a larger, grid-like composite image.
## Input types
### Required
- **`image`**
    - The input image to be tiled across the grid. It determines the visual content that will be repeated in each cell of the grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`columns`**
    - Specifies the number of columns in the grid. This determines how many times the input image is repeated horizontally.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Specifies the number of rows in the grid. This determines how many times the input image is repeated vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output composite image, consisting of the input image tiled according to the specified rows and columns.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RepeatIntoGridImage:
    """
    Tiles the input samples into a grid of configurable dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat_into_grid"
    CATEGORY = images_category_path

    def repeat_into_grid(self, image, columns, rows):
        samples = image.movedim(-1, 1)
        samples = samples.repeat(1, 1, rows, columns)
        samples = samples.movedim(1, -1)
        return (samples,)

```
