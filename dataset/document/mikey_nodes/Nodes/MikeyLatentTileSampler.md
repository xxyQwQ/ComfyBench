---
tags:
- Sampling
---

# Latent Tile Sampler (Mikey)
## Documentation
- Class name: `MikeyLatentTileSampler`
- Category: `Mikey/Sampling`
- Output node: `False`

This node specializes in resampling a latent space that exceeds the specified tile size, effectively managing larger-than-tile latent spaces by breaking them down into manageable pieces for processing. It focuses on optimizing the handling and manipulation of large latent representations for tasks such as image generation or modification, ensuring efficient and targeted resampling within a tiled framework.
## Input types
### Required
- **`base_model`**
    - Specifies the base model used for generating or refining the latent samples, serving as a critical component in the resampling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`samples`**
    - The input latent samples to be resampled, serving as the basis for the tiling and resampling operation.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`positive`**
    - The positive conditioning to guide the resampling towards desired attributes or features within the latent space.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative conditioning to steer the resampling away from certain attributes or features within the latent space.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`seed`**
    - Defines the seed for noise generation and sampling, ensuring reproducibility and consistency in the resampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - Configures the denoising factor, which can alter the characteristics of the generated or modified latent space during resampling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Determines the number of steps to be used in the sampling process, impacting the depth of exploration in the latent space.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configures the conditioning factor, which can alter the characteristics of the generated or modified latent space during resampling.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - Specifies the sampling method used for resampling, impacting the approach to navigating the latent space.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - Determines the scheduler used in conjunction with the sampler, affecting the sampling dynamics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`tile_size`**
    - Sets the size of the tiles into which the latent space is divided, directly influencing the granularity of the resampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`samples`**
    - Comfy dtype: `LATENT`
    - The resampled latent space, now segmented and processed into tiles according to the specified parameters.
    - Python dtype: `dict`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MikeyLatentTileSampler:
    # receives a latent that is larger than the tile size and resamples it
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"base_model": ("MODEL",), "samples": ("LATENT",),
                             "positive": ("CONDITIONING",), "negative": ("CONDITIONING",),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             "denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                             "steps": ("INT", {"default": 30, "min": 1, "max": 1000}),
                             "cfg": ("FLOAT", {"default": 5, "min": 0.0, "max": 1000.0, "step": 0.1}),
                             "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                             "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                             "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64})}}

    RETURN_TYPES = ('LATENT',)
    RETURN_NAMES = ('samples',)
    FUNCTION = 'sample'
    CATEGORY = 'Mikey/Sampling'

    def sample(self, seed, base_model, samples, positive, negative,
            denoise=0.25, steps=30, cfg=5, sampler_name='dpmpp_2m_sde_gpu', scheduler='karras',
            tile_size=1024):
        latent = samples.copy()
        # split latent into tiles
        latent_samples = latent["samples"]
        tiles = split_latent_tensor(latent_samples, tile_size)
        # resample each tile using ksampler
        start_step = int(steps - (steps * denoise))
        resampled_tiles = [(coords, common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, {"samples": tile}, start_step=start_step)[0]["samples"]) for coords, tile in tiles]
        # stitch the tiles to get the final upscaled latent tensor
        latent["samples"] = stitch_latent_tensors(latent_samples.shape, resampled_tiles)
        return (latent,)

```
