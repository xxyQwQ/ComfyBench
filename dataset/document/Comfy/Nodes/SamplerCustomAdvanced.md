---
tags:
- Sampling
---

# SamplerCustomAdvanced
## Documentation
- Class name: `SamplerCustomAdvanced`
- Category: `sampling/custom_sampling`
- Output node: `False`

The SamplerCustomAdvanced node is designed to provide advanced customization options for sampling processes within a generative model framework. It allows for the implementation of sophisticated sampling strategies that can be tailored to specific requirements or experimental setups, enhancing the flexibility and effectiveness of generative modeling tasks.
## Input types
### Required
- **`noise`**
    - This input represents the noise data used to perturb the sampling process, providing a means to introduce variability and explore the generative model's space more thoroughly.
    - Comfy dtype: `NOISE`
    - Python dtype: `custom.NoiseType`
- **`guider`**
    - The guider input specifies the guiding mechanism for the sampling process, such as directing the generation towards desired outcomes or away from undesired ones.
    - Comfy dtype: `GUIDER`
    - Python dtype: `custom.GuiderType`
- **`sampler`**
    - This input defines the specific sampling algorithm or strategy to be used, allowing for customization of the sampling process.
    - Comfy dtype: `SAMPLER`
    - Python dtype: `custom.SamplerType`
- **`sigmas`**
    - Sigmas represent the noise levels applied at different steps of the sampling process, enabling fine-tuned control over the introduction of variability.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`latent_image`**
    - The latent image input provides the initial state or context for the sampling process, serving as the basis for generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `custom.LatentType`
## Output types
- **`output`**
    - Comfy dtype: `LATENT`
    - This output type provides the result of the sampling process, reflecting the generated data after applying the specified sampling strategy.
    - Python dtype: `custom.LatentType`
- **`denoised_output`**
    - Comfy dtype: `LATENT`
    - The denoised output delivers the generated data with reduced noise, showcasing the effectiveness of the sampling strategy in achieving clearer, more refined results.
    - Python dtype: `custom.LatentType`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

```
