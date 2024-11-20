---
tags:
- LoRA
---

# EasyLoraStack
## Documentation
- Class name: `easy loraStack`
- Category: `EasyUse/Loaders`
- Output node: `False`

The `loraStack` node is designed to facilitate the dynamic integration and management of LoRA (Low-Rank Adaptation) layers into models. It allows for the customization of model behavior through the manipulation of LoRA parameters, enabling users to adjust the strength and impact of these layers on the model's output. This node supports the configuration of multiple LoRA layers, making it versatile for experiments with model adaptability and fine-tuning.
## Input types
### Required
- **`toggle`**
    - A boolean toggle that enables or disables the LoRA stack functionality. When enabled, it allows the node to process and apply the specified LoRA configurations.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mode`**
    - Specifies the mode of operation for the LoRA stack, such as 'simple' or 'advanced', determining how LoRA parameters are applied and managed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`num_loras`**
    - Defines the number of LoRA layers to be configured and managed by the node, allowing for dynamic adjustment based on user requirements.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_lora_stack`**
    - An optional parameter that allows for the specification of an existing LoRA stack configuration to be used or extended by the node.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `list of tuples`
- **`lora_i_name`**
    - Specifies the name of the i-th LoRA layer, enabling identification and configuration of individual layers within the stack.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_i_strength`**
    - Defines the overall strength of the i-th LoRA layer, affecting its influence on the model's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_i_model_strength`**
    - Sets the model-specific strength for the i-th LoRA layer, allowing for fine-tuned control over its impact on the model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_i_clip_strength`**
    - Determines the clip-specific strength for the i-th LoRA layer, adjusting its effect on the clip model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`lora_stack`**
    - Comfy dtype: `LORA_STACK`
    - The output of the node, which is a configured LoRA stack ready to be applied to a model. It encapsulates the LoRA layers and their parameters as specified through the node's inputs.
    - Python dtype: `list of dicts`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class loraStack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        max_lora_num = 10
        inputs = {
            "required": {
                "toggle": ("BOOLEAN", {"label_on": "enabled", "label_off": "disabled"}),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 1, "max": max_lora_num}),
            },
            "optional": {
                "optional_lora_stack": ("LORA_STACK",),
            },
        }

        for i in range(1, max_lora_num+1):
            inputs["optional"][f"lora_{i}_name"] = (
            ["None"] + folder_paths.get_filename_list("loras"), {"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_model_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = (
            "FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        return inputs

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "EasyUse/Loaders"

    def stack(self, toggle, mode, num_loras, optional_lora_stack=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return (None,)

        loras = []

        # Import Stack values
        if optional_lora_stack is not None:
            loras.extend([l for l in optional_lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        return (loras,)

```
