---
tags:
- Latent
- LatentBatch
---

# Get Latents From Batch Indexed
## Documentation
- Class name: `GetLatentsFromBatchIndexed`
- Category: `KJNodes`
- Output node: `False`

This node is designed to select and return specific latents from a given batch based on provided indices. It facilitates the extraction of a subset of latents for further processing or analysis, making it a crucial component in workflows that require manipulation or inspection of individual or groups of latents within a larger batch.
## Input types
### Required
- **`latents`**
    - Represents the batch of latents from which specific items will be selected. It is essential for determining the scope of data the node will operate on.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`indexes`**
    - A string of comma-separated indices indicating which latents to extract from the batch. This parameter allows for flexible selection of specific latents, enabling targeted manipulation or analysis.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output is a modified version of the input latent batch, containing only the latents at the specified indices.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class GetLatentsFromBatchIndexed:
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "indexedlatentsfrombatch"
    CATEGORY = "KJNodes"
    DESCRIPTION = """
Selects and returns the latents at the specified indices as an latent batch.
"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "latents": ("LATENT",),
                 "indexes": ("STRING", {"default": "0, 1, 2", "multiline": True}),
        },
    } 
    
    def indexedlatentsfrombatch(self, latents, indexes):
        
        samples = latents.copy()
        latent_samples = samples["samples"] 

        # Parse the indexes string into a list of integers
        index_list = [int(index.strip()) for index in indexes.split(',')]
        
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        
        # Select the latents at the specified indices
        chosen_latents = latent_samples[indices_tensor]

        samples["samples"] = chosen_latents
        return (samples,)

```
