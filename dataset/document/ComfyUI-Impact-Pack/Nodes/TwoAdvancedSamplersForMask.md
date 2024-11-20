---
tags:
- Sampling
---

# TwoAdvancedSamplersForMask
## Documentation
- Class name: `TwoAdvancedSamplersForMask`
- Category: `ImpactPack/Sampler`
- Output node: `False`

This node provides advanced sampling functionalities for image manipulation, specifically designed to work with masks. It enables the application of different sampling strategies to distinct regions of an image, as defined by a mask, allowing for precise control over the sampling process in areas of interest.
## Input types
### Required
- **`seed`**
    - The seed for random number generation, ensuring reproducibility of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The total number of steps to perform in the advanced sampling process, affecting the granularity of the operation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - A factor that influences the denoising process during sampling, impacting the clarity and quality of the sampled image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`samples`**
    - The initial latent images to be processed, serving as the starting point for the sampling operation.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`base_sampler`**
    - The sampler to apply to the region outside the mask, dictating the sampling behavior in non-masked areas.
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Python dtype: `object`
- **`mask_sampler`**
    - The sampler to apply to the masked region, enabling specialized sampling within the mask boundaries.
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Python dtype: `object`
- **`mask`**
    - The mask defining areas of interest for different sampling strategies, guiding the application of the base and mask samplers.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`overlap_factor`**
    - A parameter controlling the extent of mask overlap, affecting the transition between sampled regions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The modified latent image after applying advanced sampling techniques, reflecting changes in both masked and non-masked regions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class TwoAdvancedSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "samples": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "mask_sampler": ("KSAMPLER_ADVANCED", ),
                     "mask": ("MASK", ),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000})
                     },
                }

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "samples": "input latent image",
            "base_sampler": "Sampler to apply to the region outside the mask.",
            "mask_sampler": "Sampler to apply to the masked region.",
            "mask": "region mask",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask by the overlap_factor amount to overlap with other regions.",
        },
        "output": ("result latent", )
    }

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def mask_erosion(samples, mask, grow_mask_by):
        mask = mask.clone()

        w = samples['samples'].shape[3]
        h = samples['samples'].shape[2]

        mask2 = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(w, h), mode="bilinear")
        if grow_mask_by == 0:
            mask_erosion = mask2
        else:
            kernel_tensor = torch.ones((1, 1, grow_mask_by, grow_mask_by))
            padding = math.ceil((grow_mask_by - 1) / 2)

            mask_erosion = torch.clamp(torch.nn.functional.conv2d(mask2.round(), kernel_tensor, padding=padding), 0, 1)

        return mask_erosion[:, :, :w, :h].round()

    @staticmethod
    def doit(seed, steps, denoise, samples, base_sampler, mask_sampler, mask, overlap_factor):

        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        adv_steps = int(steps / denoise)
        start_at_step = adv_steps - steps

        new_latent_image = samples.copy()

        mask_erosion = TwoAdvancedSamplersForMask.mask_erosion(samples, mask, overlap_factor)

        for i in range(start_at_step, adv_steps):
            add_noise = "enable" if i == start_at_step else "disable"
            return_with_leftover_noise = "enable" if i+1 != adv_steps else "disable"

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(add_noise, seed, adv_steps, new_latent_image, i, i + 1, "enable", recovery_mode="ratio additional")

            new_latent_image['noise_mask'] = mask_erosion
            new_latent_image = mask_sampler.sample_advanced("disable", seed, adv_steps, new_latent_image, i, i + 1, return_with_leftover_noise, recovery_mode="ratio additional")

        del new_latent_image['noise_mask']

        return (new_latent_image, )

```
