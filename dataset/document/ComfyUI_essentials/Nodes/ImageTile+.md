---
tags:
- Image
- Tiled
---

# 🔧 Image Tile
## Documentation
- Class name: `ImageTile+`
- Category: `essentials/image manipulation`
- Output node: `False`

The ImageTile node is designed for slicing an image into smaller, overlapping or non-overlapping tiles based on specified rows and columns. This functionality is essential for processing large images in manageable segments or for extracting specific regions of interest with controlled overlap for detailed analysis or further manipulation.
## Input types
### Required
- **`image`**
    - The input image to be tiled. This parameter is crucial as it defines the source image from which tiles will be generated.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`rows`**
    - Specifies the number of rows to divide the image into. This affects the vertical segmentation of the image, influencing the height of each tile.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cols`**
    - Determines the number of columns to divide the image into. This parameter influences the horizontal segmentation, affecting the width of each tile.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`overlap`**
    - Defines the percentage of overlap between adjacent tiles. This parameter allows for a seamless transition between tiles, especially useful in certain analysis or reconstruction tasks.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output consists of a tensor containing the generated image tiles, ready for further processing or analysis.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ImageTile:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "cols": ("INT", { "default": 2, "min": 1, "max": 256, "step": 1, }),
                "overlap": ("FLOAT", { "default": 0, "min": 0, "max": 0.5, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image manipulation"

    def execute(self, image, rows, cols, overlap):
        h, w = image.shape[1:3]
        tile_h = h // rows
        tile_w = w // cols
        overlap_h = int(tile_h * overlap)
        overlap_w = int(tile_w * overlap)
        tile_h += overlap_h
        tile_w += overlap_w

        tiles = []
        for i in range(rows):
            for j in range(cols):
                y1 = i * tile_h
                x1 = j * tile_w

                if i > 0:
                    y1 -= overlap_h
                if j > 0:
                    x1 -= overlap_w

                y2 = y1 + tile_h
                x2 = x1 + tile_w

                if y2 > h:
                    y2 = h
                    y1 = y2 - tile_h
                if x2 > w:
                    x2 = w
                    x1 = x2 - tile_w

                tiles.append(image[:, y1:y2, x1:x2, :])

        tiles = torch.cat(tiles, dim=0)

        return(tiles,)

```
