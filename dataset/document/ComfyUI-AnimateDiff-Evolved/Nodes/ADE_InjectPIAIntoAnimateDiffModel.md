---
tags:
- AnimateDiff
- Animation
---

# 🧪Inject PIA into AnimateDiff Model 🎭🅐🅓②
## Documentation
- Class name: `ADE_InjectPIAIntoAnimateDiffModel`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA/🧪experimental`
- Output node: `False`

This node is designed to integrate PIA (Pose Independent Animation) capabilities into an existing AnimateDiff model. By injecting PIA-specific convolutional inputs into the model, it enhances the model's ability to handle motion and animation with greater flexibility and precision, particularly in scenarios where pose independence is crucial.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the motion model to be used. It is crucial for identifying the correct model within the available options, ensuring the node operates with the intended motion dynamics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`motion_model`**
    - Represents the motion model to be patched with PIA capabilities. This parameter is essential for the node to apply the necessary modifications to enable PIA features within the model.
    - Comfy dtype: `MOTION_MODEL_ADE`
    - Python dtype: `MotionModelPatcher`
### Optional
- **`ad_settings`**
    - Optional settings for AnimateDiff that can be applied during the PIA injection process. These settings allow for customization and fine-tuning of the animation and motion effects.
    - Comfy dtype: `AD_SETTINGS`
    - Python dtype: `AnimateDiffSettings`
- **`deprecation_warning`**
    - Provides a warning about the experimental status of the node, indicating that it may not function as expected.
    - Comfy dtype: `ADEWARN`
    - Python dtype: `dict`
## Output types
- **`MOTION_MODEL`**
    - Comfy dtype: `MOTION_MODEL_ADE`
    - The modified motion model with PIA capabilities injected. This output enables enhanced animation and motion handling, especially for pose-independent animations.
    - Python dtype: `MotionModelPatcher`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoadAnimateDiffAndInjectPIANode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "motion_model": ("MOTION_MODEL_ADE",),
            },
            "optional": {
                "ad_settings": ("AD_SETTINGS",),
                "deprecation_warning": ("ADEWARN", {"text": "Experimental. Don't expect to work.", "warn_type": "experimental", "color": "#CFC"}),
            }
        }
    
    RETURN_TYPES = ("MOTION_MODEL_ADE",)
    RETURN_NAMES = ("MOTION_MODEL",)

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA/🧪experimental"
    FUNCTION = "load_motion_model"
    
    def load_motion_model(self, model_name: str, motion_model: MotionModelPatcher, ad_settings: AnimateDiffSettings=None):
        # make sure model actually has PIA conv_in
        if motion_model.model.conv_in is None:
            raise Exception("Passed-in motion model was expected to be PIA (contain conv_in), but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_pia_conv_in_into_model(motion_model=loaded_motion_model, w_pia=motion_model)
        return (loaded_motion_model,)

```
