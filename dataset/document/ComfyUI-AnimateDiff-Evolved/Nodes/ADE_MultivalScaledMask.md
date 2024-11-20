---
tags:
- AnimateDiff
- Animation
---

# Multival Scaled Mask 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalScaledMask`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

The ADE_MultivalScaledMask node is designed to dynamically adjust and scale the values of a mask based on specified minimum and maximum float values. It supports both absolute and relative scaling modes, allowing for flexible manipulation of mask values to fit various animation or image processing needs.
## Input types
### Required
- **`min_float_val`**
    - Specifies the minimum value (or a list of minimum values) to which the mask's values will be scaled. This parameter sets the lower bound of the scaling operation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, List[float]]`
- **`max_float_val`**
    - Defines the maximum value (or a list of maximum values) to which the mask's values will be scaled. It sets the upper bound for the scaling, ensuring the mask's values are adjusted within a specific range.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, List[float]]`
- **`mask`**
    - The mask input represents the initial mask to be scaled. It is crucial for determining the base structure and values that will be adjusted according to the scaling parameters.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
### Optional
- **`scaling`**
    - Determines the type of scaling to be applied: absolute or relative. This choice affects how the mask values are adjusted, either by directly setting new minimum and maximum values or by normalizing them within a range.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `ScaleType`
## Output types
- **`multival`**
    - Comfy dtype: `MULTIVAL`
    - The output is a dynamically adjusted version of the input mask, with its values scaled according to the specified minimum and maximum float values and the chosen scaling mode.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MultivalScaledMaskNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "min_float_val": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.001}),
                "max_float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "mask": ("MASK",),
            },
            "optional": {
                "scaling": (ScaleType.LIST,),
            }
        }

    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, min_float_val: float, max_float_val: float, mask: Tensor, scaling: str=ScaleType.ABSOLUTE):
        lengths = [mask.shape[0]]
        iterable_inputs = [False, False]
        val_inputs = [min_float_val, max_float_val]
        if isinstance(min_float_val, Iterable):
            iterable_inputs[0] = True
            val_inputs[0] = list(min_float_val)
            lengths.append(len(min_float_val))
        if isinstance(max_float_val, Iterable):
            iterable_inputs[1] = True
            val_inputs[1] = list(max_float_val)
            lengths.append(len(max_float_val))
        # make sure mask and any iterable float_vals match max length
        max_length = max(lengths)
        mask = extend_to_batch_size(mask, max_length)
        for i in range(len(iterable_inputs)):
            if iterable_inputs[i] == True:
                # make sure tensors will match dimensions of mask
                val_inputs[i] = torch.tensor(extend_list_to_batch_size(val_inputs[i], max_length)).unsqueeze(-1).unsqueeze(-1)
        min_float_val, max_float_val = val_inputs
        if scaling == ScaleType.ABSOLUTE:
            mask = linear_conversion(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        elif scaling == ScaleType.RELATIVE:
            mask = normalize_min_max(mask.clone(), new_min=min_float_val, new_max=max_float_val)
        else:
            raise ValueError(f"scaling '{scaling}' not recognized.")
        return MultivalDynamicNode.create_multival(self, mask_optional=mask)

```
