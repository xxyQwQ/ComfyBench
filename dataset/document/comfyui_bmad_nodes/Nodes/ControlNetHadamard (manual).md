---
tags:
- ControlNet
---

# ControlNetHadamard (manual)
## Documentation
- Class name: `ControlNetHadamard (manual)`
- Category: `Bmad/conditioning`
- Output node: `False`

This node applies a control network to a set of images and conditions, adjusting the influence of the control network on the images based on a specified strength. It is designed to modify image conditioning in a batch process, where each image's conditioning is individually adjusted according to the control network's parameters and the specified strength.
## Input types
### Required
- **`conds`**
    - Conditions to be applied to each image, serving as the basis for the control network's adjustments. These conditions dictate how the images are modified in conjunction with the control network.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`control_net`**
    - The control network to be applied across the images, dictating the nature of the adjustments made to their conditioning. It plays a crucial role in how the images' conditions are modified.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNet`
- **`strength`**
    - A scalar value determining the intensity of the control network's influence on the images, allowing for fine-tuned adjustments. This value scales the effect of the control network on the conditioning.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`inputs_len`**
    - Specifies the number of images to be processed, indirectly indicating the batch size for the operation. It determines how many images the control network will be applied to.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditions for each image after the application of the control network and the specified strength. It reflects the adjusted image conditioning post-processing.
    - Python dtype: `List[Tuple[str, Dict]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ControlNetHadamardManual(ControlNetHadamard):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"conds": ("CONDITIONING",),
                             "control_net": ("CONTROL_NET",),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "inputs_len": ("INT", {"default": 9, "min": 0, "max": 32})
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = conditioning_category_path
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)

    def apply(self, conds, control_net, strength, inputs_len, **kwargs):
        inputs_len = inputs_len[0]
        images = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("image", i)
            images.append(kwargs[arg_name][0])
        return super().apply(conds, control_net, images, strength)

```
