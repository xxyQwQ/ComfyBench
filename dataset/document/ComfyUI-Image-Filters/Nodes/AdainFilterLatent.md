---
tags:
- Latent
- Normalization
---

# AdaIN Filter (Latent)
## Documentation
- Class name: `AdainFilterLatent`
- Category: `latent/filters`
- Output node: `False`

The AdainFilterLatent node applies an adaptive instance normalization (AdaIN) technique to latent representations, using a reference latent to adjust the style of the input latent. It supports adjusting the intensity of the effect through a factor and allows for varying the filter size to control the granularity of the style transfer.
## Input types
### Required
- **`latents`**
    - The input latent representations to be stylized. This is the primary data that will undergo style adjustment based on the reference latent.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`reference`**
    - The reference latent representation used to derive style characteristics that will be applied to the input latents.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`filter_size`**
    - Determines the size of the filter used in the style transfer process, affecting the granularity of the applied style.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`factor`**
    - Controls the intensity of the style transfer effect, allowing for blending between the original and stylized representations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The stylized latent representations after applying the adaptive instance normalization process.
    - Python dtype: `Tuple[Dict[str, torch.Tensor]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AdainFilterLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latents": ("LATENT", ),
                "reference": ("LATENT", ),
                "filter_size": ("INT", {"default": 1, "min": 1, "max": 128}),
                "factor": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,  "round": 0.01}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "batch_normalize"

    CATEGORY = "latent/filters"

    def batch_normalize(self, latents, reference, filter_size, factor):
        latents_copy = copy.deepcopy(latents)
        
        t = latents_copy["samples"].movedim(1, -1) # [B x C x H x W] -> [B x H x W x C]
        ref = reference["samples"].movedim(1, -1)
        
        d = filter_size * 2 + 1
        
        t_std, t_mean = std_mean_filter(t, d)
        ref_std, ref_mean = std_mean_filter(ref, d)
        
        t = (t - t_mean) / t_std
        t = t * ref_std + ref_mean
        t = t.movedim(-1, 1) # [B x H x W x C] -> [B x C x H x W]
        
        latents_copy["samples"] = torch.lerp(latents["samples"], t, factor)
        return (latents_copy,)

```
