---
tags:
- Latent
- LatentBatch
---

# Select Every Nth Latent 🎥🅥🅗🅢
## Documentation
- Class name: `VHS_SelectEveryNthLatent`
- Category: `Video Helper Suite 🎥🅥🅗🅢/latent`
- Output node: `False`

This node is designed to selectively filter latents from a given batch based on specified criteria, effectively thinning the batch to include every nth latent while optionally skipping a number of initial latents. It's particularly useful for reducing the size of a latent batch or for sampling at regular intervals within a batch.
## Input types
### Required
- **`latents`**
    - The input latent batch to be filtered. This parameter is crucial as it determines the subset of latents that will be selected based on the other criteria.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`select_every_nth`**
    - Determines the interval at which latents are selected from the batch. For example, a value of 2 would select every second latent in the batch.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`skip_first_latents`**
    - Specifies the number of initial latents to skip before beginning the selection process. This allows for more control over which latents are included in the resulting subset.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`LATENT`**
    - Comfy dtype: `LATENT`
    - The filtered subset of latents after applying the selection criteria.
    - Python dtype: `dict`
- **`count`**
    - Comfy dtype: `INT`
    - The total number of latents in the filtered subset, providing a quick reference to the size of the output batch.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SelectEveryNthLatent:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "latents": ("LATENT",),
                    "select_every_nth": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                    "skip_first_latents": ("INT", {"default": 0, "min": 0, "max": BIGMAX, "step": 1}),
                },
            }
    
    CATEGORY = "Video Helper Suite 🎥🅥🅗🅢/latent"

    RETURN_TYPES = ("LATENT", "INT",)
    RETURN_NAMES = ("LATENT", "count",)
    FUNCTION = "select_latents"

    def select_latents(self, latents: dict, select_every_nth: int, skip_first_latents: int):
        sub_latents = latents.copy()["samples"][skip_first_latents::select_every_nth]
        return ({"samples": sub_latents}, sub_latents.size(0))

```
