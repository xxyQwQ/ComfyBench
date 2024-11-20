# SplitSigmasDenoise
## Documentation
- Class name: `SplitSigmasDenoise`
- Category: `sampling/custom_sampling/sigmas`
- Output node: `False`

The SplitSigmasDenoise node is designed to partition a given sequence of sigmas into two distinct sequences based on a denoising factor. This operation facilitates the customization of the sampling process by allowing for the adjustment of the sequence length according to the desired level of denoising, thereby enabling more precise control over the generation quality.
## Input types
### Required
- **`sigmas`**
    - The sequence of sigmas to be split. It represents the noise levels at different steps of the generation process, playing a crucial role in determining the granularity of the split and the subsequent denoising effect.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`denoise`**
    - A factor determining the extent of denoising by adjusting the split point in the sigma sequence. It directly influences the balance between the high and low sigma sequences, thereby affecting the overall denoising strategy.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`high_sigmas`**
    - Comfy dtype: `SIGMAS`
    - The portion of the sigma sequence with higher values, representing the initial, less denoised stages of the generation process.
    - Python dtype: `torch.Tensor`
- **`low_sigmas`**
    - Comfy dtype: `SIGMAS`
    - The portion of the sigma sequence with lower values, indicating the later, more denoised stages of the generation process.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SplitSigmasDenoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, denoise):
        steps = max(sigmas.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        sigmas1 = sigmas[:-(total_steps)]
        sigmas2 = sigmas[-(total_steps + 1):]
        return (sigmas1, sigmas2)

```
