---
tags:
- AnimationScheduling
- Frame
- Keyframe
---

# Latent Keyframe Interpolation 🛂🅐🅒🅝
## Documentation
- Class name: `LatentKeyframeTiming`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/keyframes`
- Output node: `False`

This node is designed to manage the timing and sequencing of latent keyframes within a generative model's control network. It focuses on interpolating and scheduling keyframes based on specified timing parameters, ensuring smooth transitions and accurate timing for the generation process.
## Input types
### Required
- **`batch_index_from`**
    - Specifies the starting index for the batch of keyframes to be processed, serving as the initial point for timing and sequencing operations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`batch_index_to_excl`**
    - Defines the exclusive end index for the batch of keyframes, marking the boundary for the sequence of keyframes to be interpolated or scheduled.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`strength_from`**
    - Indicates the starting strength value for the interpolation of keyframes, setting the initial intensity or effect level.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_to`**
    - Specifies the ending strength value for the interpolation, determining the final intensity or effect level for the sequence of keyframes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`interpolation`**
    - Defines the method of interpolation to be used for transitioning between keyframes, such as linear or ease-in/out, to ensure smooth changes in strength or effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`prev_latent_kf`**
    - Optional parameter for providing a previous set of latent keyframes to be considered or integrated into the current timing and sequencing operation.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframeGroup`
- **`print_keyframes`**
    - A flag to enable logging of keyframe details for debugging or informational purposes.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`autosize`**
    - unknown
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `unknown`
## Output types
- **`LATENT_KF`**
    - Comfy dtype: `LATENT_KEYFRAME`
    - The output is a modified or newly created set of latent keyframes, reflecting the applied timing and sequencing operations.
    - Python dtype: `LatentKeyframeGroup`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LatentKeyframeInterpolationNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index_from": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
                "batch_index_to_excl": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
                "strength_from": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_to": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": (SI._LIST, ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "print_keyframes": ("BOOLEAN", {"default": False}),
                "autosize": ("ACNAUTOSIZE", {"padding": 90}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/keyframes"

    def load_keyframe(self,
                        batch_index_from: int,
                        strength_from: float,
                        batch_index_to_excl: int,
                        strength_to: float,
                        interpolation: str,
                        prev_latent_kf: LatentKeyframeGroup=None,
                        prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                        print_keyframes=False):

        if (batch_index_from > batch_index_to_excl):
            raise ValueError("batch_index_from must be less than or equal to batch_index_to.")

        if (batch_index_from < 0 and batch_index_to_excl >= 0):
            raise ValueError("batch_index_from and batch_index_to must be either both positive or both negative.")

        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        steps = batch_index_to_excl - batch_index_from
        diff = strength_to - strength_from
        if interpolation == SI.LINEAR:
            weights = np.linspace(strength_from, strength_to, steps)
        elif interpolation == SI.EASE_IN:
            index = np.linspace(0, 1, steps)
            weights = diff * np.power(index, 2) + strength_from
        elif interpolation == SI.EASE_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * (1 - np.power(1 - index, 2)) + strength_from
        elif interpolation == SI.EASE_IN_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

        for i in range(steps):
            keyframe = LatentKeyframe(batch_index_from + i, float(weights[i]))
            curr_latent_keyframe.add(keyframe)
        
        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"LatentKeyframe {keyframe.batch_index}={keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

```
