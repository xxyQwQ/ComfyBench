---
tags:
- Animation
---

# Automatic CFG - Attention modifiers
## Documentation
- Class name: `Automatic CFG - Attention modifiers`
- Category: `model_patches/Automatic_CFG/experimental_attention_modifiers`
- Output node: `False`

This node encompasses a suite of functionalities aimed at modifying and enhancing the attention mechanisms within generative models. It provides a variety of methods to adjust attention parameters dynamically, apply custom attention patches, and manipulate attention layers for improved model performance and customization. The node's capabilities allow for experimental and targeted modifications to the model's attention mechanisms, facilitating advanced configuration and optimization of generative processes.
## Input types
### Required
- **`sigma_start`**
    - Defines the starting value of sigma for the attention modification process, determining the initial scope of modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_end`**
    - Sets the ending value of sigma for the attention modification process, marking the limit of the attention adjustments.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`self_attn_mod_eval`**
    - Contains the evaluation code for self-attention modifications, allowing for dynamic and custom adjustments to the model's attention mechanism.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_input`**
    - Identifies the input UNet block IDs where the attention modifications will be applied, specifying the target areas within the model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_middle`**
    - Specifies the middle UNet block IDs for applying attention modifications, targeting intermediate layers of the model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_output`**
    - Indicates the output UNet block IDs for attention modifications, focusing on the final layers of the model.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_attn`**
    - Determines the type of attention (self, cross, or both) to be modified, guiding the application of attention adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`join_parameters`**
    - Optional parameter for joining attention modifier parameters from different sources, enhancing the flexibility of attention modifications.
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `dict`
## Output types
- **`Attention modifier`**
    - Comfy dtype: `ATTNMOD`
    - The modified attention parameters or configurations resulting from the application of attention modifiers. This output is crucial for understanding the impact of modifications on the model's attention mechanisms.
    - Python dtype: `dict`
- **`Parameters as string`**
    - Comfy dtype: `STRING`
    - A string representation of the attention modification parameters, providing a clear and concise summary of the adjustments made.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class attentionModifierParametersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "self_attn_mod_eval":   ("STRING", {"multiline": True }, {"default": ""}),
                                "unet_block_id_input":  ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_block_id_middle": ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_block_id_output": ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_attn": (["attn1","attn2","both"],),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"
    def exec(self, join_parameters=None, **kwargs):
        info_string = "\n".join([f"{k}: {v}" for k,v in kwargs.items() if v != ""])
        if kwargs['unet_attn'] == "both":
            copy_kwargs = kwargs.copy()
            kwargs['unet_attn'] = "attn1"
            copy_kwargs['unet_attn'] = "attn2"
            out_modifiers = [kwargs, copy_kwargs]
        else:
            out_modifiers = [kwargs]
        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )

```
