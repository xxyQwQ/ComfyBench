---
tags:
- GridLayout
- Image
- Tiled
---

# FlatLatentsIntoSingleGrid
## Documentation
- Class name: `FlatLatentsIntoSingleGrid`
- Category: `Bmad/latent`
- Output node: `False`

The FlatLatentsIntoSingleGrid node is designed to transform a batch of latent representations into a single, larger grid layout. This process involves rearranging individual latent samples into a grid pattern, effectively creating a single, unified latent representation that encapsulates the entire batch. This node is particularly useful in scenarios where spatial arrangement of latent samples is necessary for further processing or visualization.
## Input types
### Required
- **`latents`**
    - The 'latents' parameter represents the batch of latent representations to be transformed. It is crucial for the node's operation as it provides the raw material that will be rearranged into the grid layout. The transformation process relies on the structure and content of these latents to generate the unified grid.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a single, unified latent representation in the form of a grid. This grid is composed of the input latent samples rearranged according to the specified grid dimensions, serving as a consolidated version of the input batch for further use.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class FlatLatentsIntoSingleGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"latents": ("LATENT",), }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "flat_into_grid"
    CATEGORY = latent_category_path

    def flat_into_grid(self, latents):
        n, lc, lh, lw = latents['samples'].size()
        length_in_tiles = math.ceil(math.sqrt(n))
        new_latent = torch.zeros((1, lc, lh * math.ceil(n / length_in_tiles), lw * length_in_tiles),
                                 dtype=latents["samples"].dtype, device=latents["samples"].device)
        r = c = 0  # row and column indexes
        for i in range(n):
            x1 = x2 = lw * c
            x2 += lw
            y1 = y2 = lh * r
            y2 += lh
            new_latent[0, :, y1:y2, x1:x2] = latents["samples"][i, :, :, :]
            c += 1
            if c >= length_in_tiles:
                c = 0
                r += 1

        return ({"samples": new_latent},)

```
