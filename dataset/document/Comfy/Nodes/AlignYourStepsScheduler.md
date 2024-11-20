---
tags:
- SigmaScheduling
---

# AlignYourStepsScheduler
## Documentation
- Class name: `AlignYourStepsScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

This node is designed to adjust the noise levels (sigmas) for each step in a generative model's sampling process, based on the model type, desired number of steps, and denoising factor. It allows for fine-tuning the diffusion process to achieve more precise control over the generation quality and detail.
## Input types
### Required
- **`model_type`**
    - Specifies the type of generative model being used, allowing the scheduler to adjust noise levels appropriately. It supports a predefined set of model types.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`steps`**
    - Determines the total number of steps for the diffusion process, influencing the granularity of noise adjustment.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - A factor that adjusts the effective number of steps by scaling down the noise, enabling finer control over the diffusion process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - A sequence of noise levels (sigmas) tailored for each step of the diffusion process, enabling controlled image generation.
    - Python dtype: `torch.FloatTensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class AlignYourStepsScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_type": (["SD1", "SDXL", "SVD"], ),
                     "steps": ("INT", {"default": 10, "min": 10, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model_type, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        sigmas = NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas), )

```
