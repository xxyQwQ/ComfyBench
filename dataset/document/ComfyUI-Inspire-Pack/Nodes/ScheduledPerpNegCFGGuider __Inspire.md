# Scheduled PerpNeg CFGGuider (Inspire)
## Documentation
- Class name: `ScheduledPerpNegCFGGuider __Inspire`
- Category: `sampling/custom_sampling/guiders`
- Output node: `False`

The ScheduledPerpNegCFGGuider node dynamically adjusts the guidance scale for negative prompts in a scheduled manner during the sampling process. It integrates negative conditioning with a configurable schedule to fine-tune the influence of negative prompts over time, aiming to optimize the generation quality by balancing the guidance scale.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for generation, serving as the foundation for the sampling process.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_management.Model`
- **`positive`**
    - Defines positive conditioning prompts that guide the generation towards desired attributes or content.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`negative`**
    - Specifies negative conditioning prompts to steer the generation away from undesired attributes or content.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`empty_conditioning`**
    - Provides an option for empty conditioning, allowing for flexibility in the generation process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `str`
- **`neg_scale`**
    - Determines the scale of influence for negative prompts, allowing for fine-tuning of their impact on the generation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigmas`**
    - Specifies the noise levels for the diffusion process, contributing to the diversity of the generated content.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`from_cfg`**
    - Sets the initial guidance scale, marking the starting point of the scheduled adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`to_cfg`**
    - Defines the final guidance scale, indicating the endpoint of the scheduled adjustment.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`schedule`**
    - Determines the schedule according to which the guidance scale is adjusted over time, offering various strategies for dynamic adaptation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`guider`**
    - Comfy dtype: `GUIDER`
    - Provides the configured guider with scheduled negative conditioning, ready for use in the sampling process.
    - Python dtype: `Guider_PerpNeg_scheduled`
- **`sigmas`**
    - Comfy dtype: `SIGMAS`
    - Returns the noise levels used in the diffusion process, essential for controlling the generation's randomness.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ScheduledPerpNegCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "empty_conditioning": ("CONDITIONING", ),
                    "neg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                    "sigmas": ("SIGMAS", ),
                    "from_cfg": ("FLOAT", {"default": 6.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "to_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                    "schedule": (["linear", "log", "exp"], {'default': 'log'})
                    }
                }

    RETURN_TYPES = ("GUIDER", "SIGMAS")

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, empty_conditioning, neg_scale, sigmas, from_cfg, to_cfg, schedule):
        guider = Guider_PerpNeg_scheduled(model, sigmas, from_cfg, to_cfg, schedule, neg_scale)
        guider.set_conds(positive, negative, empty_conditioning)
        return guider, sigmas

```
