---
tags:
- SigmaScheduling
---

# GITSScheduler
## Documentation
- Class name: `GITSScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The GITSScheduler node is designed to generate a sequence of sigma values for guiding the noise levels in generative models. It adjusts the sequence based on the provided coefficients, steps, and denoise factor, offering a customizable approach to noise scheduling in sampling processes.
## Input types
### Required
- **`coeff`**
    - The coefficient value influences the overall scale of the sigma values, allowing for fine-tuning of the noise level introduced at each step of the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`steps`**
    - Defines the total number of steps in the generation process, directly impacting the length and granularity of the sigma sequence.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - A factor that adjusts the effective number of steps by reducing noise, thereby refining the output by potentially reducing the amount of introduced noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - A sequence of sigma values tailored to the specified parameters, used to control the noise level across different steps of the generative process.
    - Python dtype: `torch.FloatTensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GITSScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"coeff": ("FLOAT", {"default": 1.20, "min": 0.80, "max": 1.50, "step": 0.05}),
                     "steps": ("INT", {"default": 10, "min": 2, "max": 1000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, coeff, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        if steps <= 20:
            sigmas = NOISE_LEVELS[round(coeff, 2)][steps-2][:]
        else:
            sigmas = NOISE_LEVELS[round(coeff, 2)][-1][:]
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas), )

```
