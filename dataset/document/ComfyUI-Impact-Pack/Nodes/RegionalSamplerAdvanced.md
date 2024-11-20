---
tags:
- Sampling
---

# RegionalSamplerAdvanced
## Documentation
- Class name: `RegionalSamplerAdvanced`
- Category: `ImpactPack/Regional`
- Output node: `False`

The RegionalSamplerAdvanced node is designed to enhance image generation processes by applying region-specific sampling techniques. It allows for more precise control over the denoising and detail enhancement in different areas of an image, potentially leading to higher quality and more detailed outputs.
## Input types
### Required
- **`add_noise`**
    - Determines whether noise should be added to the sampling process, affecting the initial state of the latent image.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`noise_seed`**
    - Specifies the random seed for generating noise, ensuring reproducibility in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - Defines the total number of sampling steps to be executed, impacting the depth of the denoising process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`start_at_step`**
    - Indicates the starting step for applying the sampling, allowing for staged or delayed sampling interventions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - Marks the final step where sampling is applied, controlling the extent of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`overlap_factor`**
    - Adjusts the overlap between regions to smooth the transitions, enhancing the coherence of the sampled image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`restore_latent`**
    - Controls whether to restore the noise outside the mask area to its original state, affecting the fidelity of the inpainting.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`return_with_leftover_noise`**
    - Decides whether to return the latent image with remaining noise, influencing the final image's texture.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`latent_image`**
    - The input latent image to be processed, serving as the base for the sampling operations.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`base_sampler`**
    - The sampler used outside the region specified by the regional prompts, affecting the overall image quality.
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Python dtype: `str`
- **`regional_prompts`**
    - Prompts applied to each specified region, directing the sampling process in those areas.
    - Comfy dtype: `REGIONAL_PROMPTS`
    - Python dtype: `list`
- **`additional_mode`**
    - Specifies the operational mode for additional denoising in specific regions, offering options to either disable the feature, increase the total denoising amount, or balance the denoising between the primary and recovery samplers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`additional_sampler`**
    - Determines the recovery sampler to be used for additional denoising in specific regions, with options for automatic selection based on the primary sampler or manual selection for more customized control.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`additional_sigma_ratio`**
    - Defines the multiplier for the noise schedule in the additional denoising process, affecting how much additional denoising is applied based on the selected mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The processed latent image after applying region-specific sampling and denoising techniques.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RegionalSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "add_noise": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000}),
                     "restore_latent": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "return_with_leftover_noise": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
                     "latent_image": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "regional_prompts": ("REGIONAL_PROMPTS", ),
                     "additional_mode": (["DISABLE", "ratio additional", "ratio between"], {"default": "ratio between"}),
                     "additional_sampler": (["AUTO", "euler", "heun", "heunpp2", "dpm_2", "dpm_fast", "dpmpp_2m", "ddpm"],),
                     "additional_sigma_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                 "hidden": {"unique_id": "UNIQUE_ID"},
                }

    TOOLTIPS = {
        "input": {
            "add_noise": "Whether to add noise",
            "noise_seed": "Random seed to use for generating CPU noise for sampling.",
            "steps": "total sampling steps",
            "start_at_step": "The starting step of the sampling to be applied at this node within the range of 'steps'.",
            "end_at_step": "The step at which sampling applied at this node will stop within the range of steps (if greater than steps, sampling will continue only up to steps).",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask set in regional_prompts by the overlap_factor amount to overlap with other regions.",
            "restore_latent": "At each step, restore the noise outside the mask area to its original state, as per the principle of inpainting. This option is provided for backward compatibility, and it is recommended to always set it to true.",
            "return_with_leftover_noise": "Whether to return the latent with noise remaining if the noise has not been completely removed according to the noise schedule, or to completely remove the noise before returning it.",
            "latent_image": "input latent image",
            "base_sampler": "The sampler applied outside the area set by the regional_prompt.",
            "regional_prompts": "The prompt applied to each region",
            "additional_mode": "..._sde or uni_pc and other special samplers are used, the region is not properly denoised, and it causes a phenomenon that destroys the overall harmony. To compensate for this, a recovery operation is performed using another sampler. This requires a longer time for sampling because a second sampling is performed at each step in each region using a special sampler. 1) DISABLE: Disable this feature. 2) ratio additional: After performing the denoise amount to be performed in the step with the sampler set in the region, the recovery sampler is additionally applied by the additional_sigma_ratio. If you use this option, the total denoise amount increases by additional_sigma_ratio. 3) ratio between: The denoise amount to be performed in the step with the sampler set in the region and the denoise amount to be applied to the recovery sampler are divided by additional_sigma_ratio, and denoise is performed for each denoise amount. If you use this option, the total denoise amount does not change.",
            "additional_sampler": "1) AUTO: Automatically set the recovery sampler. If the sampler is uni_pc, uni_pc_bh2, dpmpp_sde, dpmpp_sde_gpu, the dpm_fast sampler is selected If the sampler is dpmpp_2m_sde, dpmpp_2m_sde_gpu, dpmpp_3m_sde, dpmpp_3m_sde_gpu, the dpmpp_2m sampler is selected. 2) Others: Manually set the recovery sampler.",
            "additional_sigma_ratio": "Multiplier of noise schedule to be applied according to additional_mode.",
        },
        "output": ("result latent", )
    }


    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(add_noise, noise_seed, steps, start_at_step, end_at_step, overlap_factor, restore_latent, return_with_leftover_noise, latent_image, base_sampler, regional_prompts,
             additional_mode, additional_sampler, additional_sigma_ratio, unique_id):

        if restore_latent:
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
        else:
            latent_compositor = None

        masks = [regional_prompt.mask.numpy() for regional_prompt in regional_prompts]
        masks = [np.ceil(mask).astype(np.int32) for mask in masks]
        combined_mask = torch.from_numpy(np.bitwise_or.reduce(masks))

        inv_mask = torch.where(combined_mask == 0, torch.tensor(1.0), torch.tensor(0.0))

        region_len = len(regional_prompts)
        end_at_step = min(steps, end_at_step)
        total = (end_at_step - start_at_step) * region_len

        new_latent_image = latent_image.copy()
        base_latent_image = None
        region_masks = {}

        for i in range(start_at_step, end_at_step-1):
            core.update_node_status(unique_id, f"{start_at_step+i}/{end_at_step} steps  |         ", ((i-start_at_step)*region_len)/total)

            cur_add_noise = True if i == start_at_step and add_noise else False

            if cur_add_noise:
                noise = Noise_RandomNoise(noise_seed).generate_noise(new_latent_image)
                for rp in regional_prompts:
                    noise = rp.touch_noise(noise)
            else:
                noise = None

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(cur_add_noise, noise_seed, steps, new_latent_image, i, i + 1, True,
                                                            recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio, noise=noise)

            if restore_latent:
                del new_latent_image['noise_mask']
                base_latent_image = new_latent_image.copy()

            j = 1
            for regional_prompt in regional_prompts:
                if restore_latent:
                    new_latent_image = base_latent_image.copy()

                core.update_node_status(unique_id, f"{start_at_step+i}/{end_at_step} steps  |  {j}/{region_len}", ((i-start_at_step)*region_len + j)/total)

                if j not in region_masks:
                    region_mask = regional_prompt.get_mask_erosion(overlap_factor).squeeze(0).squeeze(0)
                    region_masks[j] = region_mask
                else:
                    region_mask = region_masks[j]

                new_latent_image['noise_mask'] = region_mask
                new_latent_image = regional_prompt.sampler.sample_advanced(False, noise_seed, steps, new_latent_image, i, i + 1, True,
                                                                           recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

                if restore_latent:
                    del new_latent_image['noise_mask']
                    base_latent_image = latent_compositor.composite(base_latent_image, new_latent_image, 0, 0, False, region_mask)[0]
                    new_latent_image = base_latent_image

                j += 1

        # finalize
        core.update_node_status(unique_id, f"finalize")
        if base_latent_image is not None:
            new_latent_image = base_latent_image
        else:
            base_latent_image = new_latent_image

        new_latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample_advanced(False, noise_seed, steps, new_latent_image, end_at_step-1, end_at_step, return_with_leftover_noise,
                                                        recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

        core.update_node_status(unique_id, f"{end_at_step}/{end_at_step} steps", total)
        core.update_node_status(unique_id, "", None)

        if restore_latent:
            new_latent_image = base_latent_image

        if 'noise_mask' in new_latent_image:
            del new_latent_image['noise_mask']

        return (new_latent_image, )

```
