---
tags:
- SigmaScheduling
---

# SDTurboScheduler
## Documentation
- Class name: `SDTurboScheduler`
- Category: `sampling/custom_sampling/schedulers`
- Output node: `False`

The SDTurboScheduler node is designed to calculate a specific sequence of sigma values for diffusion models, based on the provided model, number of steps, and denoise level. It focuses on optimizing the sampling process by dynamically adjusting the timesteps according to the denoise parameter, enhancing the efficiency and quality of generated samples.
## Input types
### Required
- **`model`**
    - The model parameter specifies the diffusion model for which the sigma values are to be calculated. It is crucial for determining the appropriate sigma values based on the model's characteristics.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`steps`**
    - The steps parameter indicates the number of timesteps for which sigma values are to be calculated. It plays a key role in defining the granularity and precision of the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`denoise`**
    - The denoise parameter adjusts the starting point of the sigma calculation, allowing for finer control over the noise level in the generated samples. It directly influences the quality and characteristics of the output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - This output represents the calculated sequence of sigma values for the specified diffusion model, steps, and denoise level. It is essential for guiding the sampling process in diffusion models.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [SplitSigmas](../../Comfy/Nodes/SplitSigmas.md)



## Source code
```python
class SDTurboScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
        sigmas = model.get_model_object("model_sampling").sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas, )

```
