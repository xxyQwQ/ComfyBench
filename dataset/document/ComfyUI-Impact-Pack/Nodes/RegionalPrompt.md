---
tags:
- RegionalPrompt
---

# RegionalPrompt
## Documentation
- Class name: `RegionalPrompt`
- Category: `ImpactPack/Regional`
- Output node: `False`

The RegionalPrompt node is designed to generate regional prompts based on specified masks and advanced samplers. It allows for the creation of customized prompts that can be applied to specific regions of an image, enabling targeted modifications or enhancements. This functionality is particularly useful in scenarios where different areas of an image require distinct treatments or effects, facilitating a more granular control over the generation process.
## Input types
### Required
- **`mask`**
    - Specifies the region mask to apply the prompt to, defining the area of the image that will be affected by the generated prompt.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`advanced_sampler`**
    - Determines the advanced sampling technique to be used for the specified region, influencing how the prompt is applied and the resulting modifications.
    - Comfy dtype: `KSAMPLER_ADVANCED`
    - Python dtype: `Sampler`
### Optional
- **`variation_seed`**
    - Controls the seed for variations in the prompt generation, allowing for reproducibility or diversity in the outputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Adjusts the strength of variations applied to the prompt, enabling fine-tuning of the prompt's impact on the region.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`variation_method`**
    - Selects the method for applying variations to the prompt, such as linear or spherical linear interpolation (slerp), affecting the nature of the prompt modifications.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`regional_prompts`**
    - Comfy dtype: `REGIONAL_PROMPTS`
    - Generates regional prompts that can be utilized in the RegionalSampler, tailored to the specified regions with the applied variations.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RegionalPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "mask": ("MASK", ),
                    "advanced_sampler": ("KSAMPLER_ADVANCED", ),
                    },
                "optional": {
                    "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "variation_method": (["linear", "slerp"],),
                    }
                }

    TOOLTIPS = {
        "input": {
            "mask": "region mask",
            "advanced_sampler": "sampler for specified region",
        },
        "output": ("regional prompts. (Can be used in the RegionalSampler.)", )
    }

    RETURN_TYPES = ("REGIONAL_PROMPTS", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Regional"

    @staticmethod
    def doit(mask, advanced_sampler, variation_seed=0, variation_strength=0.0, variation_method="linear"):
        regional_prompt = core.REGIONAL_PROMPT(mask, advanced_sampler, variation_seed=variation_seed, variation_strength=variation_strength, variation_method=variation_method)
        return ([regional_prompt], )

```
