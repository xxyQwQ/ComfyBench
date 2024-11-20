# SamplerLCMDualNoise
## Documentation
- Class name: `SamplerLCMDualNoise`
- Category: `sampling/custom_sampling/samplers`
- Output node: `False`

The SamplerLCMDualNoise node provides a mechanism for generating samples using a dual noise approach within a custom sampling framework. It leverages a combination of weights, normalization steps, and the option to reuse or parallelize noise generation to enhance the sampling process.
## Input types
### Required
- **`weight`**
    - Specifies the blending weight between two noise-induced samples, influencing the balance and variation in the sampling output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`normalize_steps`**
    - Determines the number of normalization steps to apply, affecting the smoothness and quality of the generated samples.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`reuse_lcm_noise`**
    - Controls whether the same noise is reused across sampling steps, impacting the diversity and consistency of samples.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`parallel`**
    - Enables parallel processing of noise generation, potentially speeding up the sampling process.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`sampler`**
    - Comfy dtype: `SAMPLER`
    - Produces a sampler configured with dual noise characteristics for generating samples.
    - Python dtype: `comfy.samplers.KSAMPLER`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SamplerLCMDualNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "weight": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001, "round": False}),
                "normalize_steps": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "reuse_lcm_noise": ("BOOLEAN", {"default": False}),
                "parallel": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"
    FUNCTION = "get_sampler"

    def get_sampler(self, weight, normalize_steps, reuse_lcm_noise, parallel):
        return (comfy.samplers.KSAMPLER(sample_lcm_dual_noise, extra_options={"weight": weight, "normalize_steps": normalize_steps, "reuse_lcm_noise": reuse_lcm_noise, "parallel": parallel}),)

```
