---
tags:
- Sampling
---

# Latent Tile Sampler Custom (Mikey)
## Documentation
- Class name: `MikeyLatentTileSamplerCustom`
- Category: `Mikey/Sampling`
- Output node: `False`

This node is designed to handle custom operations on latent tiles within a generative model's framework. It likely extends or modifies the functionality of standard latent tile sampling processes to accommodate specific requirements or optimizations not covered by the default implementations.
## Input types
### Required
- **`model`**
    - Specifies the generative model to be used for sampling, affecting the characteristics and quality of the generated latent tiles.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`add_noise`**
    - Determines whether noise should be added to the latent tiles, influencing their variability and potentially enhancing their uniqueness.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Sets the seed for noise generation, ensuring reproducibility or variability in the added noise effects on the latent tiles.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, adjusting the influence of the conditioning on the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`positive`**
    - Positive conditioning to guide the generation towards desired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative conditioning to steer the generation away from undesired attributes or features.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`sampler`**
    - Specifies the sampling algorithm to be used, impacting the approach to navigating the latent space.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `str`
- **`sigmas`**
    - Defines the range of noise levels to be applied, affecting the diversity of the generated results.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `list`
- **`latent_image`**
    - The initial latent image to be processed, serving as the starting point for generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`tile_size`**
    - Determines the size of the tiles to be generated, affecting the granularity of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`samples`**
    - Comfy dtype: `LATENT`
    - Represents the generated latent tiles, showcasing the results of the sampling process with applied conditions and modifications.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MikeyLatentTileSamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                    "tile_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                     }
                }

    RETURN_TYPES = ("LATENT", )
    RETURN_NAMES = ("samples", )
    FUNCTION = "tile_sample"
    CATEGORY = "Mikey/Sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return out_denoised

    def tile_sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, tile_size):
        latent = latent_image.copy()
        # split image into tiles
        latent_samples = latent["samples"]
        tiles = split_latent_tensor(latent_samples, tile_size)
        # resample each tile using self.sample
        resampled_tiles = [(coords, self.sample(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, {"samples": tile})["samples"]) for coords, tile in tiles]
        # stitch the tiles to get the final upscaled latent tensor
        latent["samples"] = stitch_latent_tensors(latent_samples.shape, resampled_tiles)
        return (latent, )

```
