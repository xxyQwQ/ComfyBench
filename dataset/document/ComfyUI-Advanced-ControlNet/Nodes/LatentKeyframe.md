---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Latent Keyframe 🛂🅐🅒🅝
## Documentation
- Class name: `LatentKeyframe`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

The LatentKeyframe node is designed for creating and managing individual keyframes within a latent space. It focuses on the manipulation of specific points in the latent space, allowing for the adjustment of their strength or influence, and supports operations such as adding, replacing, and sorting keyframes based on their batch index. This node plays a crucial role in controlling the evolution of generated content by defining key moments or states within the generation process.
## Input types
### Required
- **`batch_index`**
    - Specifies the index of the keyframe within a batch. It is used to identify and manage the position of the keyframe in the sequence, enabling precise control over the timing and order of content generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength`**
    - Determines the influence or intensity of the keyframe within the latent space. Adjusting the strength allows for fine-tuning the impact of the keyframe on the generated content, enabling more nuanced control over the generation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`prev_latent_kf`**
    - An optional parameter that allows for the inclusion of previous latent keyframes, enabling the chaining or blending of keyframes for more complex generation patterns.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
- **`autosize`**
    - Configures the automatic sizing feature for the node, which adjusts the size of the input or output to fit specified requirements, enhancing flexibility in handling various data dimensions.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `ACNAUTOSIZE`
## Output types
- **`LATENT_KF`**
    - Comfy dtype: `LATENT_KEYFRAME`
    - Represents the output latent keyframe or keyframe group, encapsulating the modifications applied to the latent space through the node's operations.
    - Python dtype: `LatentKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LatentKeyframeNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                      batch_index: int,
                      strength: float,
                      prev_latent_kf: LatentKeyframeGroup=None,
                      prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                      ):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        keyframe = LatentKeyframe(batch_index, strength)
        prev_latent_keyframe.add(keyframe)
        return (prev_latent_keyframe,)

```
