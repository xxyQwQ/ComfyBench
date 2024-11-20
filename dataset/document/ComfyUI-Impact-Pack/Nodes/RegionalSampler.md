---
tags:
- Sampling
---

# RegionalSampler
## Documentation
- Class name: `RegionalSampler`
- Category: `ImpactPack/Regional`
- Output node: `False`

The RegionalSampler node is designed to selectively apply different sampling strategies across various regions of an input, enhancing the denoising process by adjusting the sampling technique based on regional characteristics. This approach aims to improve the overall harmony and quality of the generated output by compensating for areas that may not be adequately denoised using standard methods.
## Input types
### Required
- **`seed`**
    - Random seed to use for generating CPU noise for sampling, influencing the randomness of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`seed_ind`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`seed_ind_mode`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`steps`**
    - Total number of sampling steps to be performed, determining the depth of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`base_only_steps`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`denoise`**
    - The amount of noise to remove during the sampling process, affecting the clarity of the generated output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`samples`**
    - Input latent image to be processed through the sampling strategy.
    - Comfy dtype: `LATENT`
    - Python dtype: `str`
- **`base_sampler`**
    - The sampler applied outside the area set by the regional_prompt, defining the default sampling strategy.
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Python dtype: `str`
- **`regional_prompts`**
    - The prompt applied to each region, guiding the regional sampling process.
    - Comfy dtype: `REGIONAL_PROMPTS`
    - Python dtype: `str`
- **`overlap_factor`**
    - To smooth the seams of the region boundaries, expand the mask set in regional_prompts by the overlap_factor amount to overlap with other regions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`restore_latent`**
    - At each step, restore the noise outside the mask area to its original state, as per the principle of inpainting. This option is provided for backward compatibility, and it is recommended to always set it to true.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`additional_mode`**
    - Specifies the mode of operation for additional denoising in regions not properly denoised by the primary sampler. It determines whether to disable the feature, increase the total denoise amount, or divide the denoise amount between the primary and recovery samplers.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`additional_sampler`**
    - Defines the recovery sampler to be used for additional denoising. It can be set automatically based on the primary sampler or manually chosen to best suit the specific needs of the region.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`additional_sigma_ratio`**
    - Controls the multiplier for the noise schedule applied in the additional denoising step, according to the selected additional_mode.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The output latent image after processing through the regional sampling strategies.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RegionalSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "seed_2nd": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "seed_2nd_mode": (["ignore", "fixed", "seed+seed_2nd", "seed-seed_2nd", "increment", "decrement", "randomize"], ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "base_only_steps": ("INT", {"default": 2, "min": 0, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "samples": ("LATENT", ),
                     "base_sampler": ("KSAMPLER_ADVANCED", ),
                     "regional_prompts": ("REGIONAL_PROMPTS", ),
                     "overlap_factor": ("INT", {"default": 10, "min": 0, "max": 10000}),
                     "restore_latent": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "additional_mode": (["DISABLE", "ratio additional", "ratio between"], {"default": "ratio between"}),
                     "additional_sampler": (["AUTO", "euler", "heun", "heunpp2", "dpm_2", "dpm_fast", "dpmpp_2m", "ddpm"],),
                     "additional_sigma_ratio": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },
                 "hidden": {"unique_id": "UNIQUE_ID"},
                }

    TOOLTIPS = {
        "input": {
            "seed": "Random seed to use for generating CPU noise for sampling.",
            "seed_2nd": "Additional noise seed. The behavior is determined by seed_2nd_mode.",
            "seed_2nd_mode": "application method of seed_2nd. 1) ignore: Do not use seed_2nd. In the base only sampling stage, the seed is applied as a noise seed, and in the regional sampling stage, denoising is performed as it is without additional noise. 2) Others: In the base only sampling stage, the seed is applied as a noise seed, and once it is closed so that there is no leftover noise, new noise is added with seed_2nd and the regional samping stage is performed. a) fixed: Use seed_2nd as it is as an additional noise seed. b) seed+seed_2nd: Apply the value of seed+seed_2nd as an additional noise seed. c) seed-seed_2nd: Apply the value of seed-seed_2nd as an additional noise seed. d) increment: Not implemented yet. Same with fixed. e) decrement: Not implemented yet. Same with fixed. f) randomize: Not implemented yet. Same with fixed.",
            "steps": "total sampling steps",
            "base_only_steps": "total sampling steps",
            "denoise": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned.",
            "samples": "input latent image",
            "base_sampler": "The sampler applied outside the area set by the regional_prompt.",
            "regional_prompts": "The prompt applied to each region",
            "overlap_factor": "To smooth the seams of the region boundaries, expand the mask set in regional_prompts by the overlap_factor amount to overlap with other regions.",
            "restore_latent": "At each step, restore the noise outside the mask area to its original state, as per the principle of inpainting. This option is provided for backward compatibility, and it is recommended to always set it to true.",
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
    def separated_sample(*args, **kwargs):
        return separated_sample(*args, **kwargs)

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
    def doit(seed, seed_2nd, seed_2nd_mode, steps, base_only_steps, denoise, samples, base_sampler, regional_prompts, overlap_factor, restore_latent,
             additional_mode, additional_sampler, additional_sigma_ratio, unique_id=None):
        if restore_latent:
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
        else:
            latent_compositor = None

        masks = [regional_prompt.mask.numpy() for regional_prompt in regional_prompts]
        masks = [np.ceil(mask).astype(np.int32) for mask in masks]
        combined_mask = torch.from_numpy(np.bitwise_or.reduce(masks))

        inv_mask = torch.where(combined_mask == 0, torch.tensor(1.0), torch.tensor(0.0))

        adv_steps = int(steps / denoise)
        start_at_step = adv_steps - steps

        region_len = len(regional_prompts)
        total = steps*region_len

        leftover_noise = False
        if base_only_steps > 0:
            if seed_2nd_mode == 'ignore':
                leftover_noise = True

            noise = Noise_RandomNoise(seed).generate_noise(samples)

            for rp in regional_prompts:
                noise = rp.touch_noise(noise)

            samples = base_sampler.sample_advanced(True, seed, adv_steps, samples, start_at_step, start_at_step + base_only_steps, leftover_noise, recovery_mode="DISABLE", noise=noise)

        if seed_2nd_mode == "seed+seed_2nd":
            seed += seed_2nd
            if seed > 1125899906842624:
                seed = seed - 1125899906842624
        elif seed_2nd_mode == "seed-seed_2nd":
            seed -= seed_2nd
            if seed < 0:
                seed += 1125899906842624
        elif seed_2nd_mode != 'ignore':
            seed = seed_2nd

        new_latent_image = samples.copy()
        base_latent_image = None

        if not leftover_noise:
            add_noise = True
            noise = Noise_RandomNoise(seed).generate_noise(samples)

            for rp in regional_prompts:
                noise = rp.touch_noise(noise)
        else:
            add_noise = False
            noise = None

        for i in range(start_at_step+base_only_steps, adv_steps):
            core.update_node_status(unique_id, f"{i}/{steps} steps  |         ", ((i-start_at_step)*region_len)/total)

            new_latent_image['noise_mask'] = inv_mask
            new_latent_image = base_sampler.sample_advanced(add_noise, seed, adv_steps, new_latent_image, i, i + 1, True,
                                                            recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio, noise=noise)

            if restore_latent:
                if 'noise_mask' in new_latent_image:
                    del new_latent_image['noise_mask']
                base_latent_image = new_latent_image.copy()

            j = 1
            for regional_prompt in regional_prompts:
                if restore_latent:
                    new_latent_image = base_latent_image.copy()

                core.update_node_status(unique_id, f"{i}/{steps} steps  |  {j}/{region_len}", ((i-start_at_step)*region_len + j)/total)

                region_mask = regional_prompt.get_mask_erosion(overlap_factor).squeeze(0).squeeze(0)

                new_latent_image['noise_mask'] = region_mask
                new_latent_image = regional_prompt.sampler.sample_advanced(False, seed, adv_steps, new_latent_image, i, i + 1, True,
                                                                           recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

                if restore_latent:
                    del new_latent_image['noise_mask']
                    base_latent_image = latent_compositor.composite(base_latent_image, new_latent_image, 0, 0, False, region_mask)[0]
                    new_latent_image = base_latent_image

                j += 1

            add_noise = False

        # finalize
        core.update_node_status(unique_id, f"finalize")
        if base_latent_image is not None:
            new_latent_image = base_latent_image
        else:
            base_latent_image = new_latent_image

        new_latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample_advanced(False, seed, adv_steps, new_latent_image, adv_steps, adv_steps+1, False,
                                                        recovery_mode=additional_mode, recovery_sampler=additional_sampler, recovery_sigma_ratio=additional_sigma_ratio)

        core.update_node_status(unique_id, f"{steps}/{steps} steps", total)
        core.update_node_status(unique_id, "", None)

        if restore_latent:
            new_latent_image = base_latent_image

        if 'noise_mask' in new_latent_image:
            del new_latent_image['noise_mask']

        return (new_latent_image, )

```
