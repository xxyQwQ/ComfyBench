---
tags:
- GridLayout
- Image
- Tiled
---

# UnGridify (image)
## Documentation
- Class name: `UnGridify (image)`
- Category: `Bmad/image`
- Output node: `False`

The UnGridify (image) node is designed to deconstruct a grid of images into its individual tiles. It takes an image composed of a grid layout and separates it into distinct image tiles based on specified row and column parameters, facilitating the manipulation or analysis of each tile independently.
## Input types
### Required
- **`image`**
    - The composite image arranged in a grid format. This input is crucial for determining the structure from which individual tiles will be extracted.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`columns`**
    - Specifies the number of columns in the grid. This parameter is essential for accurately dividing the image into the correct number of horizontal segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Indicates the number of rows in the grid. It is vital for splitting the image into the appropriate number of vertical segments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - A list of images, where each image represents a tile extracted from the original grid. This output allows for further individual processing or analysis of each tile.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class UnGridImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "ungridify"
    CATEGORY = images_category_path
    OUTPUT_IS_LIST = (True,)

    def ungridify(self, image, columns, rows):
        tiles = []
        samples = image.movedim(-1, 1)
        _, _, height, width = samples.size()
        tile_height = height // rows
        tile_width = width // columns

        for y in range(0, rows * tile_height, tile_height):
            for x in range(0, columns * tile_width, tile_width):
                tile = samples[:, :, y:y + tile_height, x:x + tile_width]
                tile = tile.movedim(1, -1)
                tiles.append(tile)

        return (tiles,)

```
