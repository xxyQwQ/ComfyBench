# Automatic CFG - Attention modifiers tester
## Documentation
- Class name: `Automatic CFG - Attention modifiers tester`
- Category: `model_patches/Automatic_CFG/experimental_attention_modifiers`
- Output node: `False`

This node is designed to test and apply various attention modifier configurations to computational graph models, specifically focusing on enhancing or altering the attention mechanisms within these models. It aims to experiment with different attention modification strategies to optimize model performance or behavior.
## Input types
### Required
- **`seed`**
    - Specifies the seed for random number generation, ensuring reproducibility of the attention modifier application.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`sigma_start`**
    - Defines the starting value of sigma for the attention modifier, indicating the beginning of the range within which the modifier is active.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sigma_end`**
    - Defines the ending value of sigma for the attention modifier, indicating the end of the range within which the modifier is active.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`self_attn_mod_eval`**
    - Contains the evaluation code for self-attention modification, allowing for dynamic adjustment of attention mechanisms based on the model's requirements.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_input`**
    - Identifies the input block IDs where the attention modifier will be applied, enabling targeted modifications.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_middle`**
    - Identifies the middle block IDs where the attention modifier will be applied, enabling targeted modifications.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_block_id_output`**
    - Identifies the output block IDs where the attention modifier will be applied, enabling targeted modifications.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`unet_attn`**
    - Specifies the type of attention (self, cross, or both) to be modified, allowing for precise control over the modification process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
### Optional
- **`join_parameters`**
    - Allows for the combination of attention modifier parameters from another source, enhancing the flexibility of attention modification.
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `dict`
## Output types
- **`Attention modifier`**
    - Comfy dtype: `ATTNMOD`
    - Outputs the modified attention parameters, reflecting the changes made to the model's attention mechanisms.
    - Python dtype: `tuple`
- **`Parameters as string`**
    - Comfy dtype: `STRING`
    - Provides a string representation of the attention modifier parameters, facilitating easy review and documentation of the modifications.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class attentionModifierBruteforceParametersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "self_attn_mod_eval":   ("STRING", {"multiline": True , "default": ""}),
                                "unet_block_id_input":  ("STRING", {"multiline": False, "default": "4,5,7,8"}),
                                "unet_block_id_middle": ("STRING", {"multiline": False, "default": "0"}),
                                "unet_block_id_output": ("STRING", {"multiline": False, "default": "0,1,2,3,4,5"}),
                                "unet_attn": (["attn1","attn2","both"],),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def create_sequence_parameters(self, input_str, middle_str, output_str):
        input_values  = input_str.split(",")  if input_str  else []
        middle_values = middle_str.split(",") if middle_str else []
        output_values = output_str.split(",") if output_str else []
        result = []
        result.extend([{"unet_block_id_input": val, "unet_block_id_middle": "", "unet_block_id_output": ""} for val in input_values])
        result.extend([{"unet_block_id_input": "", "unet_block_id_middle": val, "unet_block_id_output": ""} for val in middle_values])
        result.extend([{"unet_block_id_input": "", "unet_block_id_middle": "", "unet_block_id_output": val} for val in output_values])
        return result

    def exec(self, seed, join_parameters=None, **kwargs):
        sequence_parameters = self.create_sequence_parameters(kwargs['unet_block_id_input'],kwargs['unet_block_id_middle'],kwargs['unet_block_id_output'])
        lenseq = len(sequence_parameters)
        current_index = seed % lenseq
        current_sequence = sequence_parameters[current_index]
        kwargs["unet_block_id_input"]  = current_sequence["unet_block_id_input"]
        kwargs["unet_block_id_middle"] = current_sequence["unet_block_id_middle"]
        kwargs["unet_block_id_output"] = current_sequence["unet_block_id_output"]
        if current_sequence["unet_block_id_input"] != "":
            current_block_string = f"unet_block_id_input: {current_sequence['unet_block_id_input']}"
        elif current_sequence["unet_block_id_middle"] != "":
            current_block_string = f"unet_block_id_middle: {current_sequence['unet_block_id_middle']}"
        elif current_sequence["unet_block_id_output"] != "":
            current_block_string = f"unet_block_id_output: {current_sequence['unet_block_id_output']}"
        info_string = f"Progress: {current_index+1}/{lenseq}\n{kwargs['self_attn_mod_eval']}\n{kwargs['unet_attn']} {current_block_string}"
        if kwargs['unet_attn'] == "both":
            copy_kwargs = kwargs.copy()
            kwargs['unet_attn'] = "attn1"
            copy_kwargs['unet_attn'] = "attn2"
            out_modifiers = [kwargs, copy_kwargs]
        else:
            out_modifiers = [kwargs]
        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )

```
