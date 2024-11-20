# GITSScheduler Func Provider
## Documentation
- Class name: `GITSSchedulerFuncProvider`
- Category: `ImpactPack/sampling`
- Output node: `False`

Provides a functionality to dynamically generate a noise schedule for image generation processes using the GITSScheduler. This node is designed to adjust the noise schedule based on given coefficients and denoise parameters, offering a flexible approach to controlling the diffusion process.
## Input types
### Required
- **`coeff`**
    - Specifies the coefficient factor for the GITSScheduler, influencing the overall noise schedule generation by adjusting the intensity of the applied noise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`denoise`**
    - Determines the amount of denoising to apply in the noise schedule, affecting the clarity and detail of the generated images.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`scheduler_func`**
    - Comfy dtype: `SCHEDULER_FUNC`
    - Returns a function capable of generating a dynamic noise schedule for use in image generation processes, tailored by the specified coefficients and denoise parameters.
    - Python dtype: `Callable`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class GITSSchedulerFuncProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "coeff": ("FLOAT", {"default": 1.20, "min": 0.80, "max": 1.50, "step": 0.05}),
                        "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }
                }

    TOOLTIPS = {
        "input": {
            "coeff": "coeff factor of GITS Scheduler",
            "denoise": "denoise amount for noise schedule",
        },
        "output": ("Returns a function that generates a noise schedule using GITSScheduler. This can be used in place of a predetermined noise schedule to dynamically generate a noise schedule based on the steps.",)
    }

    RETURN_TYPES = ("SCHEDULER_FUNC",)
    CATEGORY = "ImpactPack/sampling"

    FUNCTION = "doit"

    def doit(self, coeff, denoise):
        def f(model, sampler, steps):
            if 'GITSScheduler' not in nodes.NODE_CLASS_MAPPINGS:
                raise Exception("[Impact Pack] ComfyUI is an outdated version. Cannot use GITSScheduler.")

            scheduler = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']()
            return scheduler.get_sigmas(coeff, steps, denoise)[0]

        return (f, )

```
