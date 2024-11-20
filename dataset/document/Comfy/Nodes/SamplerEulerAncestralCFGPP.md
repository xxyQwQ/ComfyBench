# SamplerEulerAncestralCFG++
## Documentation
- Class name: `SamplerEulerAncestralCFGPP`
- Category: `sampling/custom_sampling/samplers`
- Output node: `False`

This node provides a mechanism to generate a custom sampler based on the Euler Ancestral method with specific configurations for the CFG++ algorithm. It allows for fine-tuning the sampling process by adjusting parameters related to the noise and step size, catering to specialized needs in generating samples.
## Input types
### Required
- **`eta`**
    - Specifies the step size for the Euler Ancestral CFG++ sampler, affecting the granularity of the sampling steps and the overall smoothness of the generated samples.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`s_noise`**
    - Determines the scale of noise to be added during the sampling process, influencing the diversity and variability of the generated samples.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sampler`**
    - Comfy dtype: `SAMPLER`
    - Produces a customized sampler configured for the Euler Ancestral CFG++ method, tailored for specific sampling requirements.
    - Python dtype: `comfy.samplers.KSampler`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SamplerEulerAncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step":0.01, "round": False}),
            }}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler(
            "euler_ancestral_cfg_pp",
            {"eta": eta, "s_noise": s_noise})
        return (sampler, )

```
