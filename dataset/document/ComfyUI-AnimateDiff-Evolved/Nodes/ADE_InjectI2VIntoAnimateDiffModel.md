---
tags:
- AnimateDiff
- Animation
---

# 🧪Inject I2V into AnimateDiff Model 🎭🅐🅓②
## Documentation
- Class name: `ADE_InjectI2VIntoAnimateDiffModel`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V/🧪experimental`
- Output node: `False`

This node is designed to inject an Image-to-Video (I2V) model into an existing AnimateDiff model, enhancing its capabilities with I2V features. It focuses on integrating specific I2V components into the AnimateDiff framework to enable or improve motion-based animations.
## Input types
### Required
- **`model_name`**
    - Specifies the name of the model to be loaded and modified. It is crucial for identifying the correct AnimateDiff model to which the I2V features will be added.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`motion_model`**
    - Represents the motion model patcher object that will be modified to include I2V features. It is essential for applying the I2V enhancements to the specified AnimateDiff model.
    - Comfy dtype: `MOTION_MODEL_ADE`
    - Python dtype: `MotionModelPatcher`
### Optional
- **`ad_settings`**
    - Optional settings for the AnimateDiff model that can be adjusted during the I2V injection process. These settings allow for customization and fine-tuning of the model's behavior.
    - Comfy dtype: `AD_SETTINGS`
    - Python dtype: `AnimateDiffSettings`
- **`deprecation_warning`**
    - Provides a warning message about the experimental status of this node, indicating that it may not function as expected. This parameter allows for communication of potential risks to the user.
    - Comfy dtype: `ADEWARN`
    - Python dtype: `Dict[str, Union[str, str, str]]`
## Output types
- **`MOTION_MODEL`**
    - Comfy dtype: `MOTION_MODEL_ADE`
    - The modified AnimateDiff model with I2V features injected. This output represents the enhanced model ready for use in motion-based animations.
    - Python dtype: `MotionModelPatcher`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class LoadAnimateDiffAndInjectI2VNode:
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

    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/AnimateLCM-I2V/🧪experimental"
    FUNCTION = "load_motion_model"

    def load_motion_model(self, model_name: str, motion_model: MotionModelPatcher, ad_settings: AnimateDiffSettings=None):
        # make sure model w/ encoder actually has encoder
        if motion_model.model.img_encoder is None:
            raise Exception("Passed-in motion model was expected to have an img_encoder, but did not.")
        # load motion module and motion settings, if included
        loaded_motion_model = load_motion_module_gen2(model_name=model_name, motion_model_settings=ad_settings)
        inject_img_encoder_into_model(motion_model=loaded_motion_model, w_encoder=motion_model)
        return (loaded_motion_model,)

```
