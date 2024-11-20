---
tags:
- Sampling
---

# MaskGrid N KSamplers Advanced
## Documentation
- Class name: `MaskGrid N KSamplers Advanced`
- Category: `Bmad/experimental`
- Output node: `False`

This node specializes in applying advanced sampling techniques to generate or modify latent images based on a grid pattern. It leverages mask-based operations to selectively apply changes to specific regions of the latent space, enabling complex image manipulations and enhancements.
## Input types
### Required
- **`model`**
    - The model parameter specifies the generative model used for sampling operations within the specified regions of the latent space.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`add_noise`**
    - Determines whether noise is added to the sampling process, enhancing the diversity of the generated images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`noise_seed`**
    - The seed for noise generation, ensuring reproducibility in the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`steps`**
    - The number of steps to perform in the sampling process, affecting the detail and quality of the generated images.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cfg`**
    - Configuration settings for the sampling process, allowing for customization of the generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sampler_name`**
    - The name of the sampler algorithm used for generating the latent images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scheduler`**
    - The scheduler controls the sampling process, managing the progression of steps.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `object`
- **`positive`**
    - Positive text prompts that guide the generation towards desired characteristics.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Negative text prompts that steer the generation away from undesired characteristics.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`latent_image`**
    - The latent_image parameter represents the initial state of the latent space that will undergo sampling. It is fundamental to the node's operation, serving as the canvas on which mask-based modifications are applied.
    - Comfy dtype: `LATENT`
    - Python dtype: `Dict[str, torch.Tensor]`
- **`start_at_step`**
    - The initial step of the sampling process, allowing for control over the starting point.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`end_at_step`**
    - The final step of the sampling process, defining the endpoint of generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`return_with_leftover_noise`**
    - Indicates whether to include leftover noise in the output, providing additional control over the output's characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`mask`**
    - The mask parameter is crucial for defining the regions within the latent image that will be subjected to sampling operations. It plays a key role in determining which parts of the image are modified, ensuring targeted and precise adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`rows`**
    - The number of rows in the grid pattern, defining the layout of the applied masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`columns`**
    - The number of columns in the grid pattern, further specifying the layout of the applied masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mode`**
    - The mode of operation, determining how the sampling and mask application are conducted.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`latent`**
    - Comfy dtype: `LATENT`
    - The latent output consists of a modified latent image reflecting the application of advanced sampling techniques within specified grid regions.
    - Python dtype: `Dict[str, torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class MaskGridNKSamplersAdvanced(nodes.KSamplerAdvanced):
    fork_before_sampling = {
        "Sample then Fork": False,
        "Fork then Sample": True
    }
    fork_options = list(fork_before_sampling.keys())

    @classmethod
    def INPUT_TYPES(cls):
        types = super().INPUT_TYPES()
        types["required"]["mask"] = ("IMAGE",)
        types["required"]["rows"] = ("INT", {"default": 1, "min": 1, "max": 16})
        types["required"]["columns"] = ("INT", {"default": 3, "min": 1, "max": 16})
        types["required"]["mode"] = (cls.fork_options, {"default": cls.fork_options[0]})
        return types

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "gen_batch"
    CATEGORY = f"{base_category_path}/experimental"

    def gen_batch(self, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                  latent_image, start_at_step, end_at_step, return_with_leftover_noise,
                  mask, rows, columns, mode, denoise=1.0):

        # setup sizes
        _, _, latent_height_as_img, latent_width_as_img = latent_image['samples'].size()
        latent_width_as_img *= 8
        latent_height_as_img *= 8
        _, mask_height, mask_width, _ = mask.size()

        # existing nodes required for the operation
        set_mask_node = nodes.SetLatentNoiseMask()

        latents = []

        if not self.fork_before_sampling[mode]:
            # FORK AFTER SAMPLING

            # prepare mask
            mask = RepeatIntoGridImage().repeat_into_grid(mask, columns, rows)[0]
            new_mask = torch.zeros((latent_height_as_img, latent_width_as_img))
            new_mask[:, :] = mask[0, :, :, 0]

            # prepare latent w/ mask and send to ksampler
            sampled_latent = set_mask_node.set_mask(samples=latent_image, mask=new_mask)[0]
            sampled_latent = \
            super().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
                           sampled_latent, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0][
                'samples']

            # adjust mask sizes for latent space
            mask_height //= 8
            mask_width //= 8

            # fork and copy regions from original latent
            for r in range(rows):
                for c in range(columns):
                    x2 = x1 = mask_width * c
                    x2 += mask_width
                    y2 = y1 = mask_height * r
                    y2 += mask_height
                    new_latent = latent_image['samples'].clone()
                    new_latent[0, :, y1:y2, x1:x2] = sampled_latent[0, :, y1:y2, x1:x2]
                    latents.append(new_latent)  # add new latent
        else:
            # FORK BEFORE SAMPLING
            for r in range(rows):
                for c in range(columns):
                    # copy source mask to a new empty mask
                    new_mask = torch.zeros((latent_height_as_img, latent_width_as_img))
                    new_mask[mask_height * r:mask_height * (r + 1), mask_width * c:mask_width * (c + 1)] = mask[0, :, :,
                                                                                                           0]

                    # prepare latent w/ mask and send to ksampler
                    new_latent = set_mask_node.set_mask(samples=latent_image.copy(), mask=new_mask)[0]
                    new_latent = \
                    super().sample(model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive,
                                   negative,
                                   new_latent, start_at_step, end_at_step, return_with_leftover_noise, denoise)[0][
                        'samples']

                    latents.append(new_latent)  # add new latent

        return ({"samples": torch.cat([batch for batch in latents], dim=0)},)

```
