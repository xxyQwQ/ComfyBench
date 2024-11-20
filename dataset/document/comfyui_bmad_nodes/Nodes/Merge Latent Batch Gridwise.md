---
tags:
- GridLayout
- Image
- Tiled
---

# Merge Latent Batch Gridwise
## Documentation
- Class name: `Merge Latent Batch Gridwise`
- Category: `Bmad/latent`
- Output node: `False`

This node is designed to merge a batch of latent representations into a single grid layout, utilizing specified rows and columns to organize the batch elements spatially. It aims to facilitate the visualization or further processing of batches by structuring them in a grid format, making it easier to handle and interpret the collective data.
## Input types
### Required
- **`batch`**
    - The batch of latent representations to be merged into a grid. It is the primary input that dictates the content and structure of the output grid.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`mask`**
    - A mask image used to determine the dimensions of the grid cells. Although it is primarily for fetching sizes, it plays a crucial role in defining the spatial layout of the merged grid.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`rows`**
    - Specifies the number of rows in the grid. It determines the vertical dimension of the grid layout.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`columns`**
    - Specifies the number of columns in the grid. It determines the horizontal dimension of the grid layout.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The merged latent representations arranged in a grid format. This output facilitates the collective handling and visualization of the batch.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MergeLatentsBatchGridwise:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch": ("LATENT",),
            "mask": ("IMAGE",),  # only to fetch the sizes, not really needed.
            "rows": ("INT", {"default": 1, "min": 1, "max": 16}),
            "columns": ("INT", {"default": 1, "min": 1, "max": 16})
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "merge"
    CATEGORY = latent_category_path

    def merge(self, batch, mask, rows, columns):
        _, mask_height, mask_width, _ = mask.size()
        mask_height //= 8
        mask_width //= 8
        _, cs, hs, ws = batch["samples"].size()
        print(f'{batch["samples"].size()}')
        merged = torch.empty(size=(1, cs, hs, ws), dtype=batch["samples"].dtype, device=batch["samples"].device)
        for r in range(rows):
            for c in range(columns):
                x2 = x1 = mask_width * c
                x2 += mask_width
                y2 = y1 = mask_height * r
                y2 += mask_height
                merged[0, :, y1:y2, x1:x2] = batch["samples"][c + r * columns, :, y1:y2, x1:x2]

        return ({"samples": merged},)

```
