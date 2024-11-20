---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Latent Keyframe From List 🛂🅐🅒🅝
## Documentation
- Class name: `LatentKeyframeBatchedGroup`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

This node is designed to manage and manipulate batches of latent keyframes for advanced control in generative models. It allows for the creation, modification, and interpolation of keyframes based on specified strengths and batch indices, facilitating dynamic adjustments to the generative process.
## Input types
### Required
- **`float_strengths`**
    - Defines the strengths for a batch of keyframes, enabling the creation or modification of multiple keyframes simultaneously. This parameter can accept either a single float value or an iterable of float values, allowing for flexible keyframe strength specification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, Iterable[float]]`
### Optional
- **`prev_latent_kf`**
    - An optional parameter that allows for the inclusion of previously defined latent keyframes, enabling their modification or extension with new keyframe data.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
- **`print_keyframes`**
    - A flag that, when enabled, triggers logging of keyframe details for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`autosize`**
    - An optional parameter that adjusts the size of the generated keyframes, providing flexibility in their presentation or storage.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `Dict[str, int]`
## Output types
- **`LATENT_KF`**
    - Comfy dtype: `LATENT_KEYFRAME`
    - The modified or newly created batch of latent keyframes, ready for use in further generative processes.
    - Python dtype: `LatentKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LatentKeyframeBatchedGroupNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_strengths": ("FLOAT", {"default": -1, "min": -1, "step": 0.001, "forceInput": True}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "print_keyframes": ("BOOLEAN", {"default": False}),
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self, float_strengths: Union[float, list[float]],
                      prev_latent_kf: LatentKeyframeGroup=None,
                      prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                      print_keyframes=False):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        # if received a normal float input, do nothing
        if type(float_strengths) in (float, int):
            logger.info("No batched float_strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        # if iterable, attempt to create LatentKeyframes with chosen strengths
        elif isinstance(float_strengths, Iterable):
            for idx, strength in enumerate(float_strengths):
                keyframe = LatentKeyframe(idx, strength)
                curr_latent_keyframe.add(keyframe)
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(float_strengths).__repr__}.")    

        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"LatentKeyframe {keyframe.batch_index}={keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

```
