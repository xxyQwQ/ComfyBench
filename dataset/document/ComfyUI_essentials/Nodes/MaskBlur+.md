---
tags:
- Blur
- MaskBlur
- VisualEffects
---

# 🔧 Mask Blur
## Documentation
- Class name: `MaskBlur+`
- Category: `essentials/mask`
- Output node: `False`

The `MaskBlur+` node applies a Gaussian blur to a given mask, optionally adjusting the intensity of the blur and accommodating different computational devices. It's designed to smooth out the edges or details of a mask, which can be useful in various image processing tasks where softer transitions or less pronounced features are desired.
## Input types
### Required
- **`mask`**
    - The mask to be blurred. This is the primary input on which the Gaussian blur operation is performed.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`amount`**
    - Specifies the intensity of the blur. A higher value results in a more pronounced blurring effect.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`device`**
    - Determines the computational device ('auto', 'cpu', or 'gpu') on which the blurring operation is executed, allowing for flexibility in resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`mask`**
    - Comfy dtype: `MASK`
    - The blurred version of the input mask, with potentially softened features or transitions.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - IPAdapterApply
    - [MaskToImage](../../Comfy/Nodes/MaskToImage.md)
    - [Masks Subtract](../../was-node-suite-comfyui/Nodes/Masks Subtract.md)



## Source code
```python
class MaskBlur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", { "default": 6, "min": 0, "max": 256, "step": 1, }),
                "device": (["auto", "cpu", "gpu"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "essentials/mask"

    def execute(self, mask, amount, device):
        if amount == 0:
            return (mask,)

        if "gpu" == device:
            mask = mask.to(comfy.model_management.get_torch_device())
        elif "cpu" == device:
            mask = mask.to('cpu')

        if amount % 2 == 0:
            amount+= 1

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask = T.functional.gaussian_blur(mask.unsqueeze(1), amount).squeeze(1)

        if "gpu" == device or "cpu" == device:
            mask = mask.to(comfy.model_management.intermediate_device())

        return(mask,)

```
