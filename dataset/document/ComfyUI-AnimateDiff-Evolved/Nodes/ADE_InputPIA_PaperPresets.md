# PIA Input [Paper Presets] 🎭🅐🅓②
## Documentation
- Class name: `ADE_InputPIA_PaperPresets`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA`
- Output node: `False`

This node is designed to generate input parameters for the AnimateDiff-PIA model based on predefined paper presets. It allows users to select a preset configuration, which is then used to configure the AnimateDiff-PIA model for animation generation, providing a simplified way to apply complex animation effects.
## Input types
### Required
- **`preset`**
    - Specifies the preset configuration to use for the AnimateDiff-PIA model. Each preset corresponds to a predefined set of parameters that define specific animation effects.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`batch_index`**
    - Determines the index of the batch for which the input is being generated. This is used to manage multiple animation inputs within a batch processing context.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mult_multival`**
    - An optional multiplier that can be applied to multiple values within the preset, allowing for further customization of the animation effect.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
- **`print_values`**
    - When set to True, the selected preset's parameters are logged, providing insight into the specific configuration being applied.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`pia_input`**
    - Comfy dtype: `PIA_INPUT`
    - Generates a configured input for the AnimateDiff-PIA model based on the selected paper preset.
    - Python dtype: `InputPIA`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InputPIA_PaperPresetsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preset": (PIA_RANGES._LIST_ALL,),
                "batch_index": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
            },
            "optional": {
                "mult_multival": ("MULTIVAL",),
                "print_values": ("BOOLEAN", {"default": False},),
                #"effect_multival": ("MULTIVAL",),
            }
        }

    RETURN_TYPES = ("PIA_INPUT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "create_pia_input"

    def create_pia_input(self, preset: str, batch_index: int, mult_multival: Union[float, Tensor]=None, print_values: bool=False, effect_multival: Union[float, Tensor]=None):
        # verify preset exists - function will throw error if does not
        values = PIA_RANGES.get_preset(preset)
        if print_values:
            logger.info(f"PIA Preset '{preset}': {values}")
        return (InputPIA_PaperPresets(preset=preset, index=batch_index, mult_multival=mult_multival, effect_multival=effect_multival),)

```
