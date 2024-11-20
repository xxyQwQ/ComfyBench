# SamplerLCMUpscale
## Documentation
- Class name: `SamplerLCMUpscale`
- Category: `sampling/custom_sampling/samplers`
- Output node: `False`

The SamplerLCMUpscale node is designed to upscale images using various methods, providing a flexible approach to image sampling with adjustable scale ratios and steps. It leverages a custom sampling function to enhance image resolution, catering to different upscale needs and preferences.
## Input types
### Required
- **`scale_ratio`**
    - Specifies the ratio by which the image should be upscaled. A higher value results in a larger image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`scale_steps`**
    - Determines the number of steps to perform the upscale process. A value of -1 indicates automatic determination based on the scale ratio.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`upscale_method`**
    - Selects the method used for upscaling the image, offering various algorithms to suit different quality and performance requirements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`sampler`**
    - Comfy dtype: `SAMPLER`
    - Produces a sampler configured to upscale images according to the specified parameters.
    - Python dtype: `comfy.samplers.KSAMPLER`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SamplerLCMUpscale:
    upscale_methods = ["bislerp", "nearest-exact", "bilinear", "area", "bicubic"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"scale_ratio": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 20.0, "step": 0.01}),
                     "scale_steps": ("INT", {"default": -1, "min": -1, "max": 1000, "step": 1}),
                     "upscale_method": (s.upscale_methods,),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, scale_ratio, scale_steps, upscale_method):
        if scale_steps < 0:
            scale_steps = None
        sampler = comfy.samplers.KSAMPLER(sample_lcm_upscale, extra_options={"total_upscale": scale_ratio, "upscale_steps": scale_steps, "upscale_method": upscale_method})
        return (sampler, )

```
