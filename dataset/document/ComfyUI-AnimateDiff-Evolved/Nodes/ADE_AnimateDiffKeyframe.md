---
tags:
- AnimateDiff
- Animation
---

# AnimateDiff Keyframe 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffKeyframe`
- Category: `Animate Diff 🎭🅐🅓`
- Output node: `False`

This node is designed to facilitate the creation and manipulation of keyframes within the AnimateDiff framework. It allows for the dynamic adjustment of animation parameters such as scale, effect intensity, and camera control at specific points in the animation timeline, enhancing the customization and precision of animation effects.
## Input types
### Required
- **`start_percent`**
    - Specifies the percentage point along the animation timeline at which the keyframe is to be applied, enabling precise control over when animation adjustments occur.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_ad_keyframes`**
    - An optional list of previously defined keyframes that the current keyframe will be added to, allowing for the accumulation and sequencing of animation effects.
    - Comfy dtype: `AD_KEYFRAMES`
    - Python dtype: `ADKeyframeGroup`
- **`scale_multival`**
    - Determines the scaling factor to be applied at the keyframe, affecting the size of animated elements.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
- **`effect_multival`**
    - Controls the intensity of visual effects applied at the keyframe, such as brightness or contrast adjustments.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
- **`inherit_missing`**
    - A boolean flag indicating whether the keyframe should inherit unspecified parameters from preceding keyframes, ensuring continuity in animation effects.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`guarantee_steps`**
    - Specifies the minimum number of steps for which the keyframe's effects are guaranteed to be applied, ensuring consistent animation outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`ad_keyframes`**
    - Comfy dtype: `AD_KEYFRAMES`
    - Returns an updated list of keyframes, including the newly added keyframe, facilitating the sequential application of animation effects.
    - Python dtype: `ADKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ADKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_ad_keyframes": ("AD_KEYFRAMES", ),
                "scale_multival": ("MULTIVAL",),
                "effect_multival": ("MULTIVAL",),
                "inherit_missing": ("BOOLEAN", {"default": True}, ),
                "guarantee_steps": ("INT", {"default": 1, "min": 0, "max": BIGMAX}),
            }
        }
    
    RETURN_TYPES = ("AD_KEYFRAMES", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Animate Diff 🎭🅐🅓"

    def load_keyframe(self,
                      start_percent: float, prev_ad_keyframes=None,
                      scale_multival: Union[float, torch.Tensor]=None, effect_multival: Union[float, torch.Tensor]=None,
                      cameractrl_multival: Union[float, torch.Tensor]=None, pia_input: InputPIA=None,
                      inherit_missing: bool=True, guarantee_steps: int=1):
        if not prev_ad_keyframes:
            prev_ad_keyframes = ADKeyframeGroup()
        prev_ad_keyframes = prev_ad_keyframes.clone()
        keyframe = ADKeyframe(start_percent=start_percent,
                              scale_multival=scale_multival, effect_multival=effect_multival,
                              cameractrl_multival=cameractrl_multival, pia_input=pia_input,
                              inherit_missing=inherit_missing, guarantee_steps=guarantee_steps)
        prev_ad_keyframes.add(keyframe)
        return (prev_ad_keyframes,)

```
