---
tags:
- ControlNet
---

# EasyControlnetStack
## Documentation
- Class name: `easy controlnetStack`
- Category: `EasyUse/Loaders`
- Output node: `False`

This node is designed to dynamically stack control networks for image manipulation tasks. It allows for the flexible integration of multiple control networks into a single processing pipeline, enabling advanced image conditioning and transformation based on specified control parameters. The node supports varying levels of control intensity, start and end percentages for effect application, and the ability to scale soft weights, providing a comprehensive toolkit for precise image editing.
## Input types
### Required
- **`toggle`**
    - A boolean toggle to activate or deactivate the stacking functionality. This serves as a master switch for the node's operation, determining whether the controlnet stacking process is initiated.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mode`**
    - Specifies the operation mode, which can influence the handling of start and end percentages for effect application. This allows for tailored control network stacking based on the desired level of detail and customization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`num_controlnet`**
    - Determines the number of control networks to be stacked. This parameter allows users to specify the depth of the control network stack, enabling complex image manipulations through the layering of multiple effects.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_controlnet_stack`**
    - An optional parameter for providing an existing stack of control networks. This can be used to append additional control networks to an already defined stack, enhancing flexibility in image editing workflows.
    - Comfy dtype: `CONTROL_NET_STACK`
    - Python dtype: `list`
- **`controlnet_i`**
    - Specifies the control network to be used at the ith position in the stack. This parameter allows for the dynamic selection of control networks, enabling varied and complex image manipulations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`controlnet_i_strength`**
    - Defines the intensity of the ith control network's effect. This allows for fine-tuned control over the impact of each control network within the stack, enabling subtle adjustments to the final image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent_i`**
    - Sets the starting percentage for the application of the ith control network's effect. This parameter allows for precise control over where the effect begins, enabling more targeted image manipulations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent_i`**
    - Determines the ending percentage for the application of the ith control network's effect. This parameter enables users to specify where the effect concludes, allowing for precise demarcation of the control network's influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`scale_soft_weight_i`**
    - Adjusts the scaling of soft weights for the ith control network. This parameter allows for the modulation of the control network's effect, enabling the fine-tuning of image aesthetics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`image_i`**
    - The image to be processed by the ith control network. This parameter allows for the application of control network effects to specific images, enabling customized image editing.
    - Comfy dtype: `IMAGE`
    - Python dtype: `object`
## Output types
- **`controlnet_stack`**
    - Comfy dtype: `CONTROL_NET_STACK`
    - The resulting stack of control networks after processing. This output encapsulates the combined effects of the stacked control networks, ready for application to image manipulation tasks.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class controlnetStack:

    def get_file_list(filenames):
        return [file for file in filenames if file != "put_models_here.txt" and "lllite" not in file]

    @classmethod
    def INPUT_TYPES(s):
        max_cn_num = 3
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_controlnet": ("INT", {"default": 1, "min": 1, "max": max_cn_num}),
            },
            "optional": {
                "optional_controlnet_stack": ("CONTROL_NET_STACK",),
            }
        }

        for i in range(1, max_cn_num+1):
            inputs["optional"][f"controlnet_{i}"] = (["None"] + s.get_file_list(folder_paths.get_filename_list("controlnet")), {"default": "None"})
            inputs["optional"][f"controlnet_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},)
            inputs["optional"][f"start_percent_{i}"] = ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"end_percent_{i}"] = ("FLOAT",{"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"scale_soft_weight_{i}"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001},)
            inputs["optional"][f"image_{i}"] = ("IMAGE",)
        return inputs

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("controlnet_stack",)
    FUNCTION = "stack"
    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_controlnet, optional_controlnet_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        controlnets = []

        # Import Stack values
        if optional_controlnet_stack is not None:
            controlnets.extend([l for l in optional_controlnet_stack if l[0] != "None"])

        # Import Controlnet values
        for i in range(1, num_controlnet+1):
            controlnet_name = kwargs.get(f"controlnet_{i}")

            if not controlnet_name or controlnet_name == "None":
                continue

            controlnet_strength = float(kwargs.get(f"controlnet_{i}_strength"))
            start_percent = float(kwargs.get(f"start_percent_{i}")) if mode == "advanced" else 0
            end_percent = float(kwargs.get(f"end_percent_{i}")) if mode == "advanced" else 1.0
            scale_soft_weights = float(kwargs.get(f"scale_soft_weight_{i}"))
            image = kwargs.get(f"image_{i}")

            controlnets.append((controlnet_name, controlnet_strength, start_percent, end_percent, scale_soft_weights, image, True))

        return (controlnets,)

```
