---
tags:
- Sampling
---

# TwoSamplersForMask
## Documentation
- Class name: `TwoSamplersForMask`
- Category: `ImpactPack/Sampler`
- Output node: `False`

This node is designed to apply different sampling strategies to distinct regions of a latent image, based on a specified mask. It enables the selective enhancement or alteration of image areas by utilizing separate samplers for the masked and unmasked regions, thereby facilitating targeted image manipulation within a unified framework.
## Input types
### Required
- **`latent_image`**
    - Represents the input latent image to be processed. It serves as the canvas on which different sampling strategies are applied based on the mask.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict`
- **`base_sampler`**
    - Specifies the sampler to be applied to the region outside the mask, influencing the overall appearance of the unmasked areas.
    - Comfy dtype: `KSAMPLER`
    - Python dtype: `torch.nn.Module`
- **`mask_sampler`**
    - Determines the sampler to be applied to the masked region, allowing for targeted manipulation or enhancement of specific areas.
    - Comfy dtype: `KSAMPLER`
    - Python dtype: `torch.nn.Module`
- **`mask`**
    - Defines the region to be selectively sampled or altered, guiding the application of the mask_sampler.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The resulting latent image after applying the specified sampling strategies to the masked and unmasked regions.
    - Python dtype: `Dict`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TwoSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "latent_image": ("LATENT", ),
                     "base_sampler": ("KSAMPLER", ),
                     "mask_sampler": ("KSAMPLER", ),
                     "mask": ("MASK", )
                     },
                }

    TOOLTIPS = {
        "input": {
            "latent_image": "input latent image",
            "base_sampler": "Sampler to apply to the region outside the mask.",
            "mask_sampler": "Sampler to apply to the masked region.",
            "mask": "region mask",
        },
        "output": ("result latent", )
    }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(latent_image, base_sampler, mask_sampler, mask):
        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample(latent_image)

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)

        del new_latent_image['noise_mask']

        return (new_latent_image, )

```
