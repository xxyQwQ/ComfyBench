---
tags:
- Sampling
---

# SamplerCustom
## Documentation
- Class name: `SamplerCustom`
- Category: `sampling/custom_sampling`
- Output node: `False`

The SamplerCustom node is designed to provide a customizable sampling mechanism within a broader generative modeling or simulation framework. It allows for the dynamic selection and configuration of sampling strategies to adapt to various data distributions or modeling objectives, enhancing the flexibility and efficiency of the generative process.
## Input types
### Required
- **`model`**
    - Specifies the generative model to be used for sampling, serving as the core component around which sampling strategies are configured.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`add_noise`**
    - Determines whether noise should be added to the sampling process, enhancing the diversity of generated samples.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Sets the seed for noise generation, ensuring reproducibility in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Controls the conditioning factor, adjusting the influence of specified conditions on the generated samples.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`positive`**
    - Defines positive conditioning to guide the sampling towards desired characteristics.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Specifies negative conditioning to steer the sampling away from undesired traits.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`sampler`**
    - Selects the specific sampling strategy to be employed, allowing for customization of the sampling process.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `str`
- **`sigmas`**
    - Provides control over the noise levels at different stages of the sampling process, enabling fine-tuning of the sample generation.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `list[float]`
- **`latent_image`**
    - Inputs a latent representation of an image to be used as a starting point for the sampling process.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
## Output types
- **`output`**
    - Comfy dtype: `LATENT`
    - Outputs the final sampled image in its latent representation, reflecting the applied sampling strategy and conditions.
    - Python dtype: `torch.Tensor`
- **`denoised_output`**
    - Comfy dtype: `LATENT`
    - Provides a denoised version of the sampled image, offering an alternative representation with potentially reduced noise.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [VAEDecode](../../Comfy/Nodes/VAEDecode.md)
    - [Preview Chooser](../../cg-image-picker/Nodes/Preview Chooser.md)
    - [LatentUpscaleBy](../../Comfy/Nodes/LatentUpscaleBy.md)



## Source code
```python
class SamplerCustom:
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
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

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
        return (out, out_denoised)

```
