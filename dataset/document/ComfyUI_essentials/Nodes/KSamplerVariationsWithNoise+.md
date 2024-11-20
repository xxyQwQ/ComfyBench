---
tags:
- Sampling
---

# 🔧 KSampler Variations with Noise Injection
## Documentation
- Class name: `KSamplerVariationsWithNoise+`
- Category: `essentials/sampling`
- Output node: `False`

This node introduces a two-stage sampling process with noise injection to generate variations of an input latent image. Initially, it applies a composition sampler to the input, optionally adding or fully denoising the result based on the variation strength. Subsequently, it employs a variation sampler that further processes the output of the first stage, adjusting the configuration based on the variation strength and cfg scale, to produce diverse and nuanced variations of the original image.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model used for sampling, playing a crucial role in determining the characteristics and quality of the generated variations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`latent_image`**
    - The latent_image parameter represents the input latent representation of an image, serving as the starting point for the sampling process to generate variations.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`main_seed`**
    - The main_seed parameter is used to initialize the random noise generation for the base noise, influencing the stochastic aspects of the sampling process.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
- **`steps`**
    - The steps parameter defines the number of sampling steps to be performed, affecting the detail and quality of the generated image variations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - The cfg parameter adjusts the conditioning factor, influencing the generation process by modulating the influence of the conditioning on the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The sampler_name parameter specifies the sampling strategy used in the first stage of the process, determining the approach for generating the initial variation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler parameter determines the scheduling strategy for the sampling process, affecting the progression of noise reduction and detail refinement.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive`**
    - The positive parameter provides positive textual guidance for the sampling process, steering the generation towards desired attributes or themes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - The negative parameter provides negative textual guidance, instructing the model to avoid certain attributes or themes in the generated variations.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`variation_strength`**
    - The variation_strength parameter controls the degree of variation introduced between the two stages, affecting the diversity of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_seed`**
    - The variation_seed parameter is used to seed the variation process in the second stage, ensuring reproducibility and control over the variation generation.
    - Comfy dtype: `INT:seed`
    - Python dtype: `int`
- **`denoise`**
    - The denoise parameter determines the level of denoising applied during the sampling process, affecting the clarity and detail of the generated image variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output latent represents the final varied latent representation of the image after the two-stage sampling process with noise injection.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class KSamplerVariationsWithNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "latent_image": ("LATENT", ),
                    "main_seed": ("INT:seed", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "variation_strength": ("FLOAT", {"default": 0.17, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                    #"start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    #"end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    #"return_with_leftover_noise": (["disable", "enable"], ),
                    "variation_seed": ("INT:seed", {"default": 12345, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": 0.01}),
                }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "execute"
    CATEGORY = "essentials/sampling"

    # From https://github.com/BlenderNeko/ComfyUI_Noise/
    def slerp(self, val, low, high):
        dims = low.shape

        low = low.reshape(dims[0], -1)
        high = high.reshape(dims[0], -1)

        low_norm = low/torch.norm(low, dim=1, keepdim=True)
        high_norm = high/torch.norm(high, dim=1, keepdim=True)

        low_norm[low_norm != low_norm] = 0.0
        high_norm[high_norm != high_norm] = 0.0

        omega = torch.acos((low_norm*high_norm).sum(1))
        so = torch.sin(omega)
        res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high

        return res.reshape(dims)

    def prepare_mask(self, mask, shape):
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(shape[2], shape[3]), mode="bilinear")
        mask = mask.expand((-1,shape[1],-1,-1))
        if mask.shape[0] < shape[0]:
            mask = mask.repeat((shape[0] -1) // mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        return mask

    def execute(self, model, latent_image, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, variation_strength, variation_seed, denoise):
        if main_seed == variation_seed:
            variation_seed += 1

        end_at_step = steps #min(steps, end_at_step)
        start_at_step = round(end_at_step - end_at_step * denoise)

        force_full_denoise = True
        disable_noise = True

        device = comfy.model_management.get_torch_device()

        # Generate base noise
        batch_size, _, height, width = latent_image["samples"].shape
        generator = torch.manual_seed(main_seed)
        base_noise = torch.randn((1, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).repeat(batch_size, 1, 1, 1).cpu()

        # Generate variation noise
        generator = torch.manual_seed(variation_seed)
        variation_noise = torch.randn((batch_size, 4, height, width), dtype=torch.float32, device="cpu", generator=generator).cpu()

        slerp_noise = self.slerp(variation_strength, base_noise, variation_noise)

        # Calculate sigma
        comfy.model_management.load_model_gpu(model)
        sampler = comfy.samplers.KSampler(model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=1.0, model_options=model.model_options)
        sigmas = sampler.sigmas
        sigma = sigmas[start_at_step] - sigmas[end_at_step]
        sigma /= model.model.latent_format.scale_factor
        sigma = sigma.detach().cpu().item()

        work_latent = latent_image.copy()
        work_latent["samples"] = latent_image["samples"].clone() + slerp_noise * sigma

        # if there's a mask we need to expand it to avoid artifacts, 5 pixels should be enough
        if "noise_mask" in latent_image:
            noise_mask = self.prepare_mask(latent_image["noise_mask"], latent_image['samples'].shape)
            work_latent["samples"] = noise_mask * work_latent["samples"] + (1-noise_mask) * latent_image["samples"]
            work_latent['noise_mask'] = expand_mask(latent_image["noise_mask"].clone(), 5, True)

        return common_ksampler(model, main_seed, steps, cfg, sampler_name, scheduler, positive, negative, work_latent, denoise=1.0, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)

```
