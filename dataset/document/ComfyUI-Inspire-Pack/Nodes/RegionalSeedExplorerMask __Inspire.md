---
tags:
- RegionalPrompt
---

# Regional Seed Explorer By Mask (Inspire)
## Documentation
- Class name: `RegionalSeedExplorerMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

The RegionalSeedExplorerMask node is designed to explore and apply variations to noise patterns based on seed prompts and masks. It enables the customization of generative processes in specific regions of an image, enhancing creativity and control in image synthesis.
## Input types
### Required
- **`mask`**
    - The mask parameter specifies the region of the image where the noise variations will be applied, allowing for targeted modifications.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`noise`**
    - The noise parameter represents the initial noise pattern that will be modified according to the seed prompts and mask.
    - Comfy dtype: `NOISE`
    - Python dtype: `torch.Tensor`
- **`seed_prompt`**
    - The seed_prompt parameter allows users to input specific seed prompts that guide the variation of the noise pattern in the masked region.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`enable_additional`**
    - This boolean parameter enables the inclusion of an additional seed and its strength in the variation process, offering further customization.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`additional_seed`**
    - When additional variations are enabled, this parameter specifies the additional seed to be used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`additional_strength`**
    - This parameter sets the strength of the additional seed's influence on the noise variation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - Specifies whether the noise processing should be performed on the CPU or GPU, affecting performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`variation_method`**
    - Determines the method of applying variations to the noise, such as linear, allowing for different types of noise modifications.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`noise`**
    - Comfy dtype: `NOISE`
    - The modified noise pattern after applying the seed prompts and variations according to the mask.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalSeedExplorerMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),

                "noise": ("NOISE",),
                "seed_prompt": ("STRING", {"multiline": True, "dynamicPrompts": False, "pysssss.autocomplete": False}),
                "enable_additional": ("BOOLEAN", {"default": True, "label_on": "true", "label_off": "false"}),
                "additional_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "additional_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_mode": (["GPU(=A1111)", "CPU"],),
            },
            "optional":
                {"variation_method": (["linear", "slerp"],), }
        }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(mask, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode, variation_method='linear'):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        noise = noise.to(device)
        mask = mask.to(device)

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noise.shape[2], noise.shape[3]), mode="bilinear").squeeze(0)

        try:
            seed_prompt = seed_prompt.replace("\n", "")
            items = seed_prompt.strip().split(",")

            if items == ['']:
                items = []

            if enable_additional:
                items.append((additional_seed, additional_strength))

            noise = prompt_support.SeedExplorer.apply_variation(noise, items, noise_device, mask, variation_method=variation_method)
        except Exception:
            print(f"[ERROR] IGNORED: RegionalSeedExplorerColorMask is failed.")
            traceback.print_exc()

        noise = noise.cpu()
        mask.cpu()
        return (noise,)

```
