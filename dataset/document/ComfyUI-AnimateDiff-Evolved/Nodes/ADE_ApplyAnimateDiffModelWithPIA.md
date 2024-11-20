---
tags:
- AnimateDiff
- Animation
---

# Apply AnimateDiff-PIA Model 🎭🅐🅓②
## Documentation
- Class name: `ADE_ApplyAnimateDiffModelWithPIA`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA`
- Output node: `False`

This node is designed to apply a PIA (Pose Interpolation and Animation) model within the AnimateDiff framework, enabling the animation of images by interpolating between poses. It integrates motion models with specific PIA enhancements to enrich the animation process, ensuring compatibility and leveraging PIA's capabilities for more dynamic and detailed animations.
## Input types
### Required
- **`motion_model`**
    - Specifies the motion model to be applied, which must be compatible with PIA. It is crucial for the animation process, dictating how the image is animated between poses.
    - Comfy dtype: `MOTION_MODEL_ADE`
    - Python dtype: `MotionModelPatcher`
- **`image`**
    - The input image to be animated. This image serves as the basis for the animation process, with the motion model applying transformations to it.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`vae`**
    - The VAE model used for encoding and decoding images during the animation process. It plays a key role in processing the images for animation.
    - Comfy dtype: `VAE`
    - Python dtype: `comfy.sd.VAE`
- **`start_percent`**
    - Defines the starting percentage of the animation, allowing for control over when the animation effect begins.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Specifies the ending percentage of the animation, enabling control over when the animation effect concludes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`pia_input`**
    - Optional PIA-specific input parameters that influence the animation effect. Allows for customization of the animation process.
    - Comfy dtype: `PIA_INPUT`
    - Python dtype: `InputPIA`
- **`motion_lora`**
    - A list of motion Lora settings that can be applied to the motion model for enhanced animation effects.
    - Comfy dtype: `MOTION_LORA`
    - Python dtype: `MotionLoraList`
- **`scale_multival`**
    - A multivalued parameter for scaling effects, providing additional customization for the animation.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Optional[torch.Tensor]`
- **`effect_multival`**
    - A multivalued parameter for effect intensity, offering further customization options for the animation.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Optional[torch.Tensor]`
- **`ad_keyframes`**
    - Specifies the keyframes for the animation, dictating the sequence and timing of animated poses.
    - Comfy dtype: `AD_KEYFRAMES`
    - Python dtype: `ADKeyframeGroup`
- **`prev_m_models`**
    - A list of previously applied motion models, allowing for sequential animation effects to be layered or combined.
    - Comfy dtype: `M_MODELS`
    - Python dtype: `MotionModelGroup`
## Output types
- **`m_models`**
    - Comfy dtype: `M_MODELS`
    - The updated list of motion models after applying the PIA model, with the most recent model added first.
    - Python dtype: `MotionModelGroup`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ApplyAnimateDiffPIAModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_model": ("MOTION_MODEL_ADE",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                "pia_input": ("PIA_INPUT",),
                "motion_lora": ("MOTION_LORA",),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "ad_keyframes": ("AD_KEYFRAMES",),
                "prev_m_models": ("M_MODELS",),
            }
        }

    RETURN_TYPES = ("M_MODELS",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "apply_motion_model"

    def apply_motion_model(self, motion_model: MotionModelPatcher, image: Tensor, vae: VAE,
                           start_percent: float=0.0, end_percent: float=1.0, pia_input: InputPIA=None,
                           motion_lora: MotionLoraList=None, ad_keyframes: ADKeyframeGroup=None,
                           scale_multival=None, effect_multival=None, ref_multival=None,
                           prev_m_models: MotionModelGroup=None,):
        new_m_models = ApplyAnimateDiffModelNode.apply_motion_model(self, motion_model, start_percent=start_percent, end_percent=end_percent,
                                                                    motion_lora=motion_lora, ad_keyframes=ad_keyframes,
                                                                    scale_multival=scale_multival, effect_multival=effect_multival, prev_m_models=prev_m_models)
        # most recent added model will always be first in list;
        curr_model = new_m_models[0].models[0]
        # confirm that model is PIA
        if curr_model.model.mm_info.mm_format != AnimateDiffFormat.PIA:
            raise Exception(f"Motion model '{curr_model.model.mm_info.mm_name}' is not a PIA model; cannot be used with Apply AnimateDiff-PIA Model node.")
        curr_model.orig_pia_images = image
        curr_model.pia_vae = vae
        if pia_input is None:
            pia_input = InputPIA_Multival(1.0)
        curr_model.pia_input = pia_input
        #curr_model.pia_multival = ref_multival
        return new_m_models

```
