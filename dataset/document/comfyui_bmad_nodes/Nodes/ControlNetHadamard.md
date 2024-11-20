---
tags:
- ControlNet
---

# ControlNetHadamard
## Documentation
- Class name: `ControlNetHadamard`
- Category: `Bmad/conditioning`
- Output node: `False`

This node applies a control network to a set of images and conditions, modulating the latter based on the control network's output and a specified strength. It's designed to integrate control network effects into image conditioning, allowing for dynamic adjustments to the conditioning process.
## Input types
### Required
- **`conds`**
    - The conditions to be modulated by the control network, representing the contextual or semantic guidance for image generation.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Dict[str, Any]]`
- **`control_net`**
    - The control network to be applied, determining the nature and extent of modulation on the conditions.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `torch.nn.Module`
- **`image`**
    - The images to which the control network adjustments are applied, serving as the basis for condition modulation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `List[torch.Tensor]`
- **`strength`**
    - A scalar value determining the intensity of the control network's effect on the conditions, allowing for fine-tuned control over the modulation.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modulated conditions, adjusted by the control network to reflect the desired changes.
    - Python dtype: `List[Dict[str, Any]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ControlNetHadamard(nodes.ControlNetApply):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conds": ("CONDITIONING",),
                             "control_net": ("CONTROL_NET",),
                             "image": ("IMAGE",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = conditioning_category_path
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def apply(self, conds, control_net, images, strength):
        control_net = control_net[0]
        strength = strength[0]

        assert len(images) == len(conds), "lists sizes do not match"

        print(len(images))
        print(len(images[0]))
        print(len(conds))
        new_conds = []
        for i in range(len(images)):
            new_conds.append(super().apply_controlnet(conds[i], control_net, images[i], strength)[0])
        return (new_conds,)

```
