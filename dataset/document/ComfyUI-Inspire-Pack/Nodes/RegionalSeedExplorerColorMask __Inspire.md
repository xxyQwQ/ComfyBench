---
tags:
- RegionalPrompt
---

# Regional Seed Explorer By Color Mask (Inspire)
## Documentation
- Class name: `RegionalSeedExplorerColorMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in exploring seed variations within specific regions of an image, guided by a color mask. It applies modifications to the noise pattern based on seed prompts and additional parameters, allowing for targeted adjustments and enhancements in image generation processes.
## Input types
### Required
- **`color_mask`**
    - The color mask used to define regions of interest within the image. It serves as a guide for applying seed variations, enabling precise control over where modifications occur.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask_color`**
    - The specific color in the color mask that identifies the regions of interest. This color acts as a key to isolate parts of the image for seed exploration.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`noise`**
    - The noise pattern to be modified. It represents the base canvas on which seed variations are applied, influencing the final image output.
    - Comfy dtype: `NOISE`
    - Python dtype: `torch.Tensor`
- **`seed_prompt`**
    - A comma-separated list of seeds and their corresponding prompts. These are used to guide the variation process, dictating how and where adjustments are made.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`enable_additional`**
    - A flag indicating whether additional seed and strength parameters should be considered in the variation process. When true, it allows for further customization of the seed exploration.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`additional_seed`**
    - An additional seed value to be included in the seed exploration process. It offers an extra layer of customization, working alongside the main seed prompts.
    - Comfy dtype: `INT`
    - Python dtype: `str`
- **`additional_strength`**
    - The strength of the additional seed's influence on the variation process. It determines the impact of the additional seed on the final image modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`noise_mode`**
    - Specifies the computational device (CPU or GPU) for processing the noise pattern. This choice affects performance and efficiency during the seed exploration.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`variation_method`**
    - The method used to apply variations to the noise pattern. It defines the algorithmic approach for integrating seed prompts and additional parameters into the image generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`noise`**
    - Comfy dtype: `NOISE`
    - The modified noise pattern after applying seed variations. It reflects the targeted adjustments made based on the input prompts and parameters.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The mask used to define regions of interest within the image, guiding where the seed variations are applied.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalSeedExplorerColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),

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

    RETURN_TYPES = ("NOISE", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(color_mask, mask_color, noise, seed_prompt, enable_additional, additional_seed, additional_strength, noise_mode, variation_method='linear'):
        device = comfy.model_management.get_torch_device()
        noise_device = "cpu" if noise_mode == "CPU" else device

        color_mask = color_mask.to(device)
        noise = noise.to(device)

        mask = color_to_mask(color_mask, mask_color)
        original_mask = mask
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(noise.shape[2], noise.shape[3]), mode="bilinear").squeeze(0)

        mask = mask.to(device)

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

        color_mask.cpu()
        noise = noise.cpu()
        original_mask = original_mask.cpu()
        return (noise, original_mask)

```
