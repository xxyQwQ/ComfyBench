---
tags:
- LatentNoise
- Noise
---

# 🔧 Remove Latent Mask
## Documentation
- Class name: `RemoveLatentMask+`
- Category: `essentials/utilities`
- Output node: `False`

This node is designed to remove the noise mask from a given set of latent samples. It ensures that the latent samples are cleaned of any previously applied noise masks, maintaining the integrity of the original latent representation.
## Input types
### Required
- **`samples`**
    - The latent samples from which the noise mask is to be removed. This operation is crucial for processes that require the original, unaltered state of the latent samples.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The cleaned latent samples, with the noise mask removed, ready for further processing or generation tasks.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RemoveLatentMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT",),}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"

    CATEGORY = "essentials/utilities"

    def execute(self, samples):
        s = samples.copy()
        if "noise_mask" in s:
            del s["noise_mask"]

        return (s,)

```
