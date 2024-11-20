---
tags:
- AnimateDiff
- Animation
---

# Multival Dynamic 🎭🅐🅓
## Documentation
- Class name: `ADE_MultivalDynamic`
- Category: `Animate Diff 🎭🅐🅓/multival`
- Output node: `False`

The ADE_MultivalDynamic node is designed to dynamically handle multiple values within the AnimateDiff framework, enabling the flexible manipulation and application of various parameters across different aspects of the animation and diffusion process.
## Input types
### Required
- **`float_val`**
    - This parameter accepts a single float or a list of floats, representing the dynamic values to be manipulated within the AnimateDiff framework. It is crucial for defining the intensity or magnitude of effects applied in the animation process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `Union[float, list[float]]`
### Optional
- **`mask_optional`**
    - An optional mask input that allows for targeted application of effects, enabling precise control over which areas of the animation are affected by the specified dynamic values.
    - Comfy dtype: `MASK`
    - Python dtype: `Tensor`
## Output types
- **`multival`**
    - Comfy dtype: `MULTIVAL`
    - This output is a dynamic representation of multiple values, tailored to influence various aspects of the animation and diffusion process within the AnimateDiff framework.
    - Python dtype: `Multival`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ADE_ApplyAnimateDiffModel](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_ApplyAnimateDiffModel.md)



## Source code
```python
class MultivalDynamicNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "float_val": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001},),
            },
            "optional": {
                "mask_optional": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("MULTIVAL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/multival"
    FUNCTION = "create_multival"

    def create_multival(self, float_val: Union[float, list[float]]=1.0, mask_optional: Tensor=None):
        # first, normalize inputs
        # if float_val is iterable, treat as a list and assume inputs are floats
        float_is_iterable = False
        if isinstance(float_val, Iterable):
            float_is_iterable = True
            float_val = list(float_val)
            # if mask present, make sure float_val list can be applied to list - match lengths
            if mask_optional is not None:
                if len(float_val) < mask_optional.shape[0]:
                    # copies last entry enough times to match mask shape
                    float_val = extend_list_to_batch_size(float_val, mask_optional.shape[0])
                if mask_optional.shape[0] < len(float_val):
                    mask_optional = extend_to_batch_size(mask_optional, len(float_val))
                float_val = float_val[:mask_optional.shape[0]]
            float_val: Tensor = torch.tensor(float_val).unsqueeze(-1).unsqueeze(-1)
        # now that inputs are normalized, figure out what value to actually return
        if mask_optional is not None:
            mask_optional = mask_optional.clone()
            if float_is_iterable:
                mask_optional = mask_optional[:] * float_val.to(mask_optional.dtype).to(mask_optional.device)
            else:
                mask_optional = mask_optional * float_val
            return (mask_optional,)
        else:
            if not float_is_iterable:
                return (float_val,)
            # create a dummy mask of b,h,w=float_len,1,1 (sigle pixel)
            # purpose is for float input to work with mask code, without special cases
            float_len = float_val.shape[0] if float_is_iterable else 1
            shape = (float_len,1,1)
            mask_optional = torch.ones(shape)
            mask_optional = mask_optional[:] * float_val.to(mask_optional.dtype).to(mask_optional.device)
            return (mask_optional,)

```
