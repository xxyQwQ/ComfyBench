---
tags:
- AnimateDiff
- Animation
---

# Sample Settings 🎭🅐🅓
## Documentation
- Class name: `ADE_AnimateDiffSamplingSettings`
- Category: `Animate Diff 🎭🅐🅓`
- Output node: `False`

This node is designed to configure and manage the sampling settings for the AnimateDiff process, allowing for detailed customization of the animation generation process. It enables users to specify various parameters such as batch offsets, noise types, seed generation methods, and more, to finely tune the animation output according to specific requirements.
## Input types
### Required
- **`batch_offset`**
    - Specifies the offset for the batch processing, allowing for control over the starting point of animation generation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`noise_type`**
    - Defines the type of noise to be applied during the sampling process, influencing the visual texture and quality of the generated animation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_gen`**
    - Determines the method for generating seeds used in the animation process, impacting the randomness and variation of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed_offset`**
    - Sets the offset for seed generation, providing additional control over the randomness of the animation.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`noise_layers`**
    - Optional parameter that allows for the specification of custom noise layers, offering further customization of the animation's visual effects.
    - Comfy dtype: `NOISE_LAYERS`
    - Python dtype: `NoiseLayerGroup`
- **`iteration_opts`**
    - Provides options for iteration control during the sampling process, enabling adjustments to the animation's progression and detail.
    - Comfy dtype: `ITERATION_OPTS`
    - Python dtype: `IterationOptions`
- **`seed_override`**
    - Allows for a specific seed value to be used, overriding the default or generated seed for precise control over the animation outcome.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`adapt_denoise_steps`**
    - Enables or disables the adaptation of denoise steps based on specific conditions, affecting the smoothness and quality of the animation.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`custom_cfg`**
    - Permits the inclusion of custom configuration settings for keyframes, enhancing the animation's narrative and visual coherence.
    - Comfy dtype: `CUSTOM_CFG`
    - Python dtype: `CustomCFGKeyframeGroup`
- **`sigma_schedule`**
    - Specifies a sigma schedule to control the variance of noise throughout the animation, impacting the overall visual dynamics.
    - Comfy dtype: `SIGMA_SCHEDULE`
    - Python dtype: `SigmaSchedule`
- **`image_inject`**
    - Allows for the injection of specific images into the animation, adding complexity and detail to the visual output.
    - Comfy dtype: `IMAGE_INJECT`
    - Python dtype: `NoisedImageToInjectGroup`
## Output types
- **`settings`**
    - Comfy dtype: `SAMPLE_SETTINGS`
    - Returns the configured sampling settings, encapsulating all specified parameters and adjustments for the animation process.
    - Python dtype: `SampleSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SampleSettingsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_offset": ("INT", {"default": 0, "min": 0, "max": BIGMAX}),
                "noise_type": (NoiseLayerType.LIST,),
                "seed_gen": (SeedNoiseGeneration.LIST,),
                "seed_offset": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX}),
            },
            "optional": {
                "noise_layers": ("NOISE_LAYERS",),
                "iteration_opts": ("ITERATION_OPTS",),
                "seed_override": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "adapt_denoise_steps": ("BOOLEAN", {"default": False},),
                "custom_cfg": ("CUSTOM_CFG",),
                "sigma_schedule": ("SIGMA_SCHEDULE",),
                "image_inject": ("IMAGE_INJECT",),
            }
        }

    RETURN_TYPES = ("SAMPLE_SETTINGS",)
    RETURN_NAMES = ("settings",)
    CATEGORY = "Animate Diff 🎭🅐🅓"
    FUNCTION = "create_settings"

    def create_settings(self, batch_offset: int, noise_type: str, seed_gen: str, seed_offset: int, noise_layers: NoiseLayerGroup=None,
                        iteration_opts: IterationOptions=None, seed_override: int=None, adapt_denoise_steps=False,
                        custom_cfg: CustomCFGKeyframeGroup=None, sigma_schedule: SigmaSchedule=None, image_inject: NoisedImageToInjectGroup=None):
        sampling_settings = SampleSettings(batch_offset=batch_offset, noise_type=noise_type, seed_gen=seed_gen, seed_offset=seed_offset, noise_layers=noise_layers,
                                           iteration_opts=iteration_opts, seed_override=seed_override, adapt_denoise_steps=adapt_denoise_steps,
                                           custom_cfg=custom_cfg, sigma_schedule=sigma_schedule, image_injection=image_inject)
        return (sampling_settings,)

```
