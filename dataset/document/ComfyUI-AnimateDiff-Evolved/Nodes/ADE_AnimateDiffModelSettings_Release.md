---
tags:
- AnimateDiff
- Animation
---

# 🚫[DEPR] Motion Model Settings 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettings_Release`
- Category: ``
- Output node: `False`

This node is designed to configure motion model settings for the AnimateDiff process, allowing users to adjust motion scale parameters to fine-tune the animation effects.
## Input types
### Required
- **`min_motion_scale`**
    - Specifies the minimum scale for motion in the animation, serving as a baseline for how subtle or pronounced the motion effects should be.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scale for motion, setting an upper limit on the intensity of the animation effects.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask_motion_scale`**
    - An optional mask tensor to apply scale adjustments selectively across different parts of the image, enhancing the control over motion effects.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - Returns the configured motion model settings, encapsulating the adjustments made to motion scales.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
                "optional": {"deprecation_warning": ("ADEWARN", {"text": "Deprecated"})},
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = ""  #"Animate Diff 🎭🅐🅓/① Gen1 nodes ①/motion settings"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        motion_model_settings = AnimateDiffSettings(
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
            )

        return (motion_model_settings,)

```
