---
tags:
- LoRA
- LoRAVisualization
---

# XY Inputs: Lora //EasyUse
## Documentation
- Class name: `easy XYInputs: Lora`
- Category: `EasyUse/XY Inputs`
- Output node: `False`

This node is designed to facilitate the generation and manipulation of XY plot values based on Lora configurations. It dynamically adjusts model and clip strengths for each Lora, optionally incorporating weights and allowing for the extension of Lora configurations through an optional stack. The node aims to provide a flexible interface for visualizing and customizing Lora-based data representations.
## Input types
### Required
- **`input_mode`**
    - Specifies the mode of input, determining whether weights are included in the Lora configurations. It affects how the node processes and generates XY plot values.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_count`**
    - Defines the number of Loras to be considered for generating XY plot values. It directly influences the size and complexity of the output data.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`model_strength`**
    - Sets a default or specific model strength for each Lora, used as a baseline unless overridden by individual Lora configurations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clip_strength`**
    - Determines a default or specific clip strength for each Lora, serving as a baseline unless individual Lora configurations specify otherwise.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_name_i`**
    - Specifies the name of each Lora, allowing for individual identification and configuration within the XY plot generation process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model_str_i`**
    - Sets the model strength for each Lora individually, overriding the default model strength if specified.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clip_str_i`**
    - Sets the clip strength for each Lora individually, overriding the default clip strength if specified.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`optional_lora_stack`**
    - Allows for the extension of Lora configurations beyond the specified count, enabling more complex and customized data representations.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `list`
## Output types
- **`X or Y`**
    - Comfy dtype: `X_Y`
    - Represents the generated XY plot values based on the provided Lora configurations, ready for visualization or further processing.
    - Python dtype: `dict`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class XYplot_Lora:

    modes = ["Lora Names", "Lora Names+Weights"]

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")

        inputs = {
            "required": {
                "input_mode": (cls.modes,),
                "lora_count": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

        for i in range(1, 10 + 1):
            inputs["required"][f"lora_name_{i}"] = (loras,)
            inputs["required"][f"model_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["required"][f"clip_str_{i}"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})

        inputs["optional"] = {
            "optional_lora_stack": ("LORA_STACK",)
        }
        return inputs

    RETURN_TYPES = ("X_Y",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"

    CATEGORY = "EasyUse/XY Inputs"

    def xy_value(self, input_mode, lora_count, model_strength, clip_strength, **kwargs):

        axis = "advanced: Lora"
        # Extract values from kwargs
        loras = [kwargs.get(f"lora_name_{i}") for i in range(1, lora_count + 1)]
        model_strs = [kwargs.get(f"model_str_{i}", model_strength) for i in range(1, lora_count + 1)]
        clip_strs = [kwargs.get(f"clip_str_{i}", clip_strength) for i in range(1, lora_count + 1)]

        # Use model_strength and clip_strength for the loras where values are not provided
        if "Weights" not in input_mode:
            for i in range(lora_count):
                model_strs[i] = model_strength
                clip_strs[i] = clip_strength

        # Extend each sub-array with lora_stack if it's not None
        values = [lora.replace(',', '*')+','+str(model_str)+','+str(clip_str) for lora, model_str, clip_str
                    in zip(loras, model_strs, clip_strs) if lora != "None"]

        optional_lora_stack = kwargs.get("optional_lora_stack") if "optional_lora_stack" in kwargs else []

        xy_values = {"axis": axis, "values": values, "lora_stack": optional_lora_stack}
        return (xy_values,)

```
