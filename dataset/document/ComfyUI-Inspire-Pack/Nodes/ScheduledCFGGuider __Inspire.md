# Scheduled CFGGuider (Inspire)
## Documentation
- Class name: `ScheduledCFGGuider __Inspire`
- Category: `sampling/custom_sampling/guiders`
- Output node: `False`

The ScheduledCFGGuider node in the Inspire pack is designed to dynamically adjust the guidance scale during the sampling process based on a predefined schedule. It allows for the customization of the guidance process by scheduling the guidance scale from a starting value to an ending value over the course of the sampling, enhancing the control over the generation quality and diversity.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for sampling, serving as the foundation for the guidance process. The choice of model directly influences the effectiveness and direction of the guidance applied.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.samplers.CFGGuider`
- **`positive`**
    - Defines positive conditioning to guide the model towards generating content that aligns with specified conditions. Positive conditioning influences the model to favor certain attributes or themes in the generated output.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`negative`**
    - Specifies negative conditioning to steer the model away from generating content that matches these conditions. Negative conditioning helps in avoiding undesired elements or themes in the generated content.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`sigmas`**
    - Represents the noise levels used in the sampling process, affecting the exploration of the generative space. The variation in noise levels can significantly impact the diversity and novelty of the generated outputs.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `list[float]`
- **`from_cfg`**
    - The starting guidance scale value, initiating the dynamic adjustment of guidance during sampling. This value sets the initial intensity of guidance, impacting the starting point of the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`to_cfg`**
    - The ending guidance scale value, concluding the dynamic adjustment of guidance. The final guidance scale value determines the intensity of guidance towards the end of the sampling process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`schedule`**
    - The schedule type for adjusting the guidance scale, such as linear, logarithmic, or exponential, providing flexibility in the guidance process. The choice of schedule affects how the guidance intensity changes over time.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`guider`**
    - Comfy dtype: `GUIDER`
    - The configured guider object, ready to guide the sampling process according to the specified conditions and schedule. It encapsulates the logic for dynamically adjusting guidance based on the provided schedule.
    - Python dtype: `Guider_scheduled`
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - The noise levels used in the sampling process, returned for further use or analysis. These levels are crucial for understanding and controlling the exploration within the generative model's space.
    - Python dtype: `list[float]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ScheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sigmas": ("SIGMAS", ),
                    "from_cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "to_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "schedule": (["linear", "log", "exp"], {'default': 'log'})
                    }
                }

    RETURN_TYPES = ("GUIDER", "SIGMAS")

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, sigmas, from_cfg, to_cfg, schedule):
        guider = Guider_scheduled(model, sigmas, from_cfg, to_cfg, schedule)
        guider.set_conds(positive, negative)
        return guider, sigmas

```
