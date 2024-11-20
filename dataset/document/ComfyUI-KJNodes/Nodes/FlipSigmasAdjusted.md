---
tags:
- SigmaScheduling
---

# Flip Sigmas Adjusted
## Documentation
- Class name: `FlipSigmasAdjusted`
- Category: `KJNodes/noise`
- Output node: `False`

The FlipSigmasAdjusted node is designed to manipulate a sequence of sigma values for use in diffusion models. It reverses the order of the sigma values, applies an offset, and optionally divides the entire sequence by the last sigma value or another specified divisor. This process is crucial for adjusting the noise levels in a controlled manner, ensuring that the diffusion process can be fine-tuned according to specific requirements.
## Input types
### Required
- **`sigmas`**
    - The sequence of sigma values to be adjusted. This sequence is reversed and potentially modified to meet certain conditions, such as avoiding zero values and applying offsets or divisions.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`divide_by_last_sigma`**
    - A boolean flag indicating whether the sigma values should be divided by the last sigma value in the sequence. This is used to normalize the sequence based on its final value.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`divide_by`**
    - A scalar value by which the entire sequence of sigma values is divided. This allows for further adjustment of the noise level across the sequence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`offset_by`**
    - An integer indicating how many positions to offset the sigma values after reversing their order. This can be used to shift the noise levels applied at each step of the diffusion process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`SIGMAS`**
    - Comfy dtype: `SIGMAS`
    - The adjusted sequence of sigma values, ready for use in diffusion processes. This output is a tensor of sigma values that have been reversed, offset, and potentially normalized or divided as specified.
    - Python dtype: `torch.Tensor`
- **`sigmas_string`**
    - Comfy dtype: `STRING`
    - A string representation of the adjusted sigma values, formatted with a specified precision and separator for easy readability.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FlipSigmasAdjusted:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     "divide_by_last_sigma": ("BOOLEAN", {"default": False}),
                     "divide_by": ("FLOAT", {"default": 1,"min": 1, "max": 255, "step": 0.01}),
                     "offset_by": ("INT", {"default": 1,"min": -100, "max": 100, "step": 1}),
                     }
                }
    RETURN_TYPES = ("SIGMAS", "STRING",)
    RETURN_NAMES = ("SIGMAS", "sigmas_string",)
    CATEGORY = "KJNodes/noise"
    FUNCTION = "get_sigmas_adjusted"

    def get_sigmas_adjusted(self, sigmas, divide_by_last_sigma, divide_by, offset_by):
        
        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        adjusted_sigmas = sigmas.clone()
        #offset sigma
        for i in range(1, len(sigmas)):
            offset_index = i - offset_by
            if 0 <= offset_index < len(sigmas):
                adjusted_sigmas[i] = sigmas[offset_index]
            else:
                adjusted_sigmas[i] = 0.0001 
        if adjusted_sigmas[0] == 0:
            adjusted_sigmas[0] = 0.0001  
        if divide_by_last_sigma:
            adjusted_sigmas = adjusted_sigmas / adjusted_sigmas[-1]

        sigma_np_array = adjusted_sigmas.numpy()
        array_string = np.array2string(sigma_np_array, precision=2, separator=', ', threshold=np.inf)
        adjusted_sigmas = adjusted_sigmas / divide_by
        return (adjusted_sigmas, array_string,)

```
