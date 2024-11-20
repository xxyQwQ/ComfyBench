# Multival Dynamic [Float List] 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalDynamicFloatInput`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

This node specializes in dynamically generating multivalued outputs based on floating-point inputs, optionally incorporating a mask. It is designed to facilitate the creation of nuanced, variable outputs in animation and differential processing contexts, where the precise control and modulation of values can significantly impact the resulting animations or transformations.
## Input types
### Required
- **`float_val`**
    - A primary floating-point value or a list of such values, serving as the core input for generating multivalued outputs. This parameter allows for fine-grained control over the output generation process, enabling customization and variability.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, list[float]]`
### Optional
- **`mask_optional`**
    - An optional mask input that can be used to modulate or restrict the application of the float_val across different areas, providing an additional layer of control and customization. It should be a tensor of a specific shape and data type, typically related to image processing masks.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor[float]`
## Output types
- **`multival`**
    - Comfy dtype: `MULTIVAL`
    - The output is a dynamically generated multivalued entity, tailored according to the provided floating-point values and optional mask, suitable for varied applications in animation and differential processing.
    - Python dtype: `MultivalDynamicNode`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MultivalDynamicFloatInputNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001, "forceInput": True},),
            },
            "optional": {
                "mask_optional": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=None, mask_optional: Tensor=None):
        return MultivalDynamicNode.create_multival(self, float_val=float_val, mask_optional=mask_optional)

```
