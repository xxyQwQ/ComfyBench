---
tags:
- Noise
---

# RandomNoise (inspire)
## Documentation
- Class name: `RandomNoise __Inspire`
- Category: `InspirePack/a1111_compat`
- Output node: `False`

The RandomNoise __Inspire node is designed to generate and apply random noise to input data, leveraging various parameters to control the noise characteristics. This node supports customization of noise generation through seeds, modes, and variation methods, enabling diverse effects on the data for creative or experimental purposes.
## Input types
### Required
- **`noise_seed`**
    - Specifies the initial seed for noise generation, serving as the basis for the random noise process. This seed ensures reproducibility of the noise patterns across runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_mode`**
    - Determines the computational device (CPU or GPU) used for noise generation, affecting performance and compatibility with different hardware setups.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`batch_seed_mode`**
    - Controls how seeds are incremented across batches, offering various strategies for introducing variability or consistency in noise generation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`variation_seed`**
    - Defines a separate seed for controlling noise variation, allowing for additional randomness or pattern changes in the noise applied.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`variation_strength`**
    - Sets the intensity of the noise variation, adjusting how much the noise alters the input data.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`variation_method`**
    - Specifies the method used to blend the original and varied noise, influencing the final noise effect on the data.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`noise`**
    - Comfy dtype: `NOISE`
    - Produces a modified version of the input data with applied random noise, reflecting the specified parameters and variation effects.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RandomNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "noise_mode": (["GPU(=A1111)", "CPU"],),
                    "batch_seed_mode": (["incremental", "comfy", "variation str inc:0.01", "variation str inc:0.05"],),
                    "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    },
                "optional":
                    {"variation_method": (["linear", "slerp"],), }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "InspirePack/a1111_compat"

    def get_noise(self, noise_seed, noise_mode, batch_seed_mode, variation_seed, variation_strength, variation_method="linear"):
        return (Inspire_RandomNoise(noise_seed, noise_mode, batch_seed_mode, variation_seed, variation_strength, variation_method=variation_method),)

```
