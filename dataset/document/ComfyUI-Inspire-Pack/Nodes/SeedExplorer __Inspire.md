---
tags:
- RandomGeneration
- Seed
---

# Seed Explorer (Inspire)
## Documentation
- Class name: `SeedExplorer __Inspire`
- Category: `InspirePack/Prompt`
- Output node: `False`

The SeedExplorer node is designed to explore and manipulate seed values within the Inspire Pack ecosystem. It provides functionality to adjust, set, or randomize seeds used in various generative processes, facilitating controlled variability and reproducibility in generated content.
## Input types
### Required
- **`latent`**
    - Specifies the latent space vector to be used or modified in the seed exploration process, serving as a foundation for generative variations.
    - Comfy dtype: `LATENT`
    - Python dtype: `torch.Tensor`
- **`seed_prompt`**
    - A string input that can be used to influence the seed exploration process, potentially guiding the generation towards specific themes or directions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`enable_additional`**
    - A boolean flag that enables or disables the use of additional seed manipulation features, offering more granular control over the generation process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`additional_seed`**
    - An integer representing an additional seed value to be used in conjunction with the primary seed, allowing for further customization of the generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`additional_strength`**
    - A float indicating the strength of the influence of the additional seed on the generation process, providing a means to adjust the impact of seed variations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - Specifies the computational backend (GPU or CPU) for noise generation, affecting the performance and possibly the outcomes of the seed exploration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`initial_batch_seed_mode`**
    - Determines the method for initializing seeds in batch processes, influencing the diversity and consistency of generated batches.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`variation_method`**
    - Defines the method for applying variations to the seed or latent space, such as linear interpolation or spherical linear interpolation (slerp), affecting the nature of generative changes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model`**
    - Identifies the model to be used in conjunction with the seed exploration, linking the seed manipulation to specific generative models.
    - Comfy dtype: `model`
    - Python dtype: `str`
## Output types
- **`noise`**
    - Comfy dtype: `NOISE`
    - The modified or newly generated noise vector resulting from the seed exploration process, ready for further generative applications.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SeedExplorer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
                "initial_batch_seed_mode": (["incremental", "comfy"],),
            },
            "optional":
                {
                    "variation_method": (["linear", "slerp"],),
                    "model": ("model",),
                }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Prompt"

    @staticmethod
    def apply_variation(start_noise, seed_items, noise_device, mask=None, variation_method='linear'):
        noise = start_noise
        for x in seed_items:
            if isinstance(x, str):
                item = x.split(':')
            else:
                item = x

            if len(item) == 2:
                try:
                    variation_seed = int(item[0])
                    variation_strength = float(item[1])

                    noise = utils.apply_variation_noise(noise, noise_device, variation_seed, variation_strength, mask=mask, variation_method=variation_method)
                except Exception:
                    print(f"[ERROR] IGNORED: SeedExplorer failed to processing '{x}'")
                    traceback.print_exc()
        return noise

    @staticmethod
    def doit(latent, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode,
             initial_batch_seed_mode, variation_method='linear', model=None):
        latent_image = latent["samples"]

        if hasattr(comfy.sample, 'fix_empty_latent_channels') and model is not None:
            latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        seed_prompt = seed_prompt.replace("\n", "")
        items = seed_prompt.strip().split(",")

        if items == ['']:
            items = []

        if enable_additional:
            items.append((additional_seed, additional_strength))

        try:
            hd = items[0]
            tl = items[1:]

            if isinstance(hd, tuple):
                hd_seed = int(hd[0])
            else:
                hd_seed = int(hd)

            noise = utils.prepare_noise(latent_image, hd_seed, None, noise_device, initial_batch_seed_mode)
            noise = noise.to(device)
            noise = SeedExplorer.apply_variation(noise, tl, noise_device, variation_method=variation_method)
            noise = noise.cpu()

            return (noise,)

        except Exception:
            print(f"[ERROR] IGNORED: SeedExplorer failed")
            traceback.print_exc()

        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout,
                            device=noise_device)
        return (noise,)

```
