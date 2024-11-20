# PIA Input [Multival] 🎭🅐🅓②
## Documentation
- Class name: `ADE_InputPIA_Multival`
- Category: `Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA`
- Output node: `False`

The node is designed to handle multiple values for PIA (Perceptual Image Annotation) inputs, allowing for the customization and fine-tuning of animation effects in the AnimateDiff framework. It facilitates the integration of varied intensity levels or parameters to influence the animation outcome.
## Input types
### Required
- **`multival`**
    - Specifies the intensity level or parameter value for the PIA input, affecting the animation's visual effects.
    - Comfy dtype: `MULTIVAL`
    - Python dtype: `Union[float, torch.Tensor]`
## Output types
- **`pia_input`**
    - Comfy dtype: `PIA_INPUT`
    - Produces a PIA input object configured with the specified multival parameter, ready for use in animation processing.
    - Python dtype: `InputPIA`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InputPIA_MultivalNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multival": ("MULTIVAL",),
            },
            # "optional": {
            #     "effect_multival": ("MULTIVAL",),
            # }
        }
    
    RETURN_TYPES = ("PIA_INPUT",)
    CATEGORY = "Animate Diff 🎭🅐🅓/② Gen2 nodes ②/PIA"
    FUNCTION = "create_pia_input"

    def create_pia_input(self, multival: Union[float, Tensor], effect_multival: Union[float, Tensor]=None):
        return (InputPIA_Multival(multival, effect_multival),)

```
