---
tags:
- GridLayout
- Image
- Tiled
---

# Repeat Into Grid (latent)
## Documentation
- Class name: `Repeat Into Grid (latent)`
- Category: `Bmad/latent`
- Output node: `False`

This node is designed to replicate and arrange input latent samples into a grid format based on specified dimensions. It enables the creation of a structured layout of latent representations, facilitating operations that require uniform spatial arrangements, such as visualization or further processing in grid-based models.
## Input types
### Required
- **`samples`**
    - The latent samples to be tiled into a grid. This input is crucial for determining the content that will be replicated across the grid.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`columns`**
    - Specifies the number of columns in the grid. This parameter directly influences the grid's width and the arrangement of the replicated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rows`**
    - Determines the number of rows in the grid. It affects the grid's height and how the samples are distributed vertically.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a modified version of the input latent samples, now arranged into a grid as specified by the input dimensions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RepeatIntoGridLatent:
    """
    Tiles the input samples into a grid of configurable dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",),
                             "columns": grid_len_INPUT,
                             "rows": grid_len_INPUT,
                             }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "repeat_into_grid"
    CATEGORY = latent_category_path

    def repeat_into_grid(self, samples, columns, rows):
        s = samples.copy()
        samples = samples['samples']
        tiled_samples = samples.repeat(1, 1, rows, columns)
        s['samples'] = tiled_samples
        return (s,)

```
