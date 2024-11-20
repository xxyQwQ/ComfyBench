---
tags:
- Animation
---

# Automatic CFG - Excellent attention
## Documentation
- Class name: `Automatic CFG - Excellent attention`
- Category: `model_patches/Automatic_CFG`
- Output node: `False`

This node specializes in dynamically configuring a model's attention mechanism to enhance its performance. It applies a series of attention modifiers based on various conditions, including positive, negative, and global modifiers, to optimize the model's focus and processing capabilities. This customization allows for improved effectiveness in generating or interpreting data by adjusting the model's attention in response to specific scenarios.
## Input types
### Required
- **`model`**
    - The model to which dynamic attention configuration modifications will be applied, serving as the foundation for optimizing attention mechanisms.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`Auto_CFG`**
    - A flag indicating whether automatic configuration is enabled, allowing the node to dynamically adjust attention mechanisms based on predefined conditions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`patch_multiplier`**
    - A multiplier that adjusts the intensity of the attention modification, influencing the strength of the applied attention patches.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`patch_cond`**
    - A condition that triggers the application of attention modifiers designed to enhance the model's focus under specific scenarios.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`patch_uncond`**
    - A condition that triggers the application of attention modifiers regardless of specific scenarios, applying a uniform enhancement across the model.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`light_patch`**
    - A flag that determines the application of lighter, less intensive attention modifiers, offering a subtler adjustment to the model's attention mechanism.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mute_self_input_layer_i_cond`**
    - A condition that mutes the self-attention inputs at a specified layer under certain conditions, reducing the model's internal focus at that layer.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mute_cross_input_layer_i_cond`**
    - A condition that mutes the cross-attention inputs at a specified layer under certain conditions, reducing the model's external focus at that layer.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mute_self_input_layer_i_uncond`**
    - A condition that unconditionally mutes the self-attention inputs at a specified layer, reducing the model's internal focus at that layer regardless of external conditions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`mute_cross_input_layer_i_uncond`**
    - A condition that unconditionally mutes the cross-attention inputs at a specified layer, reducing the model's external focus at that layer regardless of external conditions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`uncond_sigma_end`**
    - The ending value of sigma for unconditional attention modification, defining the lower bound of the modification's intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`bypass_layer_i_instead_of_mute`**
    - A flag that, when enabled, bypasses the attention mechanism at a specified layer instead of muting it, allowing for alternative processing pathways.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`save_as_preset`**
    - A flag indicating whether to save the configured attention modifications as a preset for future use, facilitating easy replication of the configuration.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`preset_name`**
    - The name under which the attention modification preset will be saved, allowing for easy identification and reuse.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
### Optional
- **`attn_mod_for_positive_operation`**
    - Additional attention modifiers to be applied for positive operations, enhancing the model's focus on relevant aspects.
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `list`
- **`attn_mod_for_negative_operation`**
    - Additional attention modifiers to be applied for negative operations, mitigating the model's focus on less relevant aspects.
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `list`
## Output types
- **`Model`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
- **`Parameters as string`**
    - Comfy dtype: `STRING`
    - A string representation of the parameters used for the attention modifications, providing insight into the configuration applied.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFGExcellentattentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {"required": {
                        "model": ("MODEL",),
                        "Auto_CFG": ("BOOLEAN", {"default": True}),
                        "patch_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0, "round": 0.01}),
                        "patch_cond":   ("BOOLEAN", {"default": True}),
                        "patch_uncond": ("BOOLEAN", {"default": True}),
                        "light_patch":  ("BOOLEAN", {"default": False}),
                        "mute_self_input_layer_8_cond":    ("BOOLEAN", {"default": False}),
                        "mute_cross_input_layer_8_cond":   ("BOOLEAN", {"default": False}),
                        "mute_self_input_layer_8_uncond":  ("BOOLEAN", {"default": True}),
                        "mute_cross_input_layer_8_uncond": ("BOOLEAN", {"default": False}),
                        "uncond_sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                        "bypass_layer_8_instead_of_mute": ("BOOLEAN", {"default": False}),
                        "save_as_preset": ("BOOLEAN", {"default": False}),
                        "preset_name":    ("STRING",  {"multiline": False}),
                        },
                        "optional":{
                                "attn_mod_for_positive_operation": ("ATTNMOD", {"forceInput": True}),
                                "attn_mod_for_negative_operation": ("ATTNMOD", {"forceInput": True}),
                        },
                    }
        if "dev_env.txt" in os.listdir(current_dir):
            inputs['optional'].update({"attn_mod_for_global_operation": ("ATTNMOD", {"forceInput": True})})
        return inputs
    
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model", "Parameters as string",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, Auto_CFG, patch_multiplier, patch_cond, patch_uncond, light_patch,
              mute_self_input_layer_8_cond, mute_cross_input_layer_8_cond,
              mute_self_input_layer_8_uncond, mute_cross_input_layer_8_uncond,
              uncond_sigma_end,bypass_layer_8_instead_of_mute, save_as_preset, preset_name,
              attn_mod_for_positive_operation = None, attn_mod_for_negative_operation = None, attn_mod_for_global_operation = None):
        
        parameters_as_string = "Excellent attention:\n" + "\n".join([f"{k}: {v}" for k, v in locals().items() if k not in ["self", "model"]])
        
        with open(os.path.join(json_preset_path, "Excellent_attention.json"), 'r', encoding='utf-8') as f:
            patch_parameters = json.load(f)

        attn_patch = {"sigma_start": 1000, "sigma_end": 0,
                       "self_attn_mod_eval": f"normalize_tensor(q+(q-attention_basic(attnbc, k, v, extra_options['n_heads'])))*attnbc.norm()*{patch_multiplier}",
                       "unet_block_id_input": "", "unet_block_id_middle": "0", "unet_block_id_output": "", "unet_attn": "attn2"}
        attn_patch_light = {"sigma_start": 1000, "sigma_end": 0,
                       "self_attn_mod_eval": f"q*{patch_multiplier}",
                       "unet_block_id_input": "", "unet_block_id_middle": "0", "unet_block_id_output": "", "unet_attn": "attn2"}

        kill_self_input_8 = {
            "sigma_start": 1000,
            "sigma_end": 0,
            "self_attn_mod_eval": "q" if bypass_layer_8_instead_of_mute else "torch.zeros_like(q)",
            "unet_block_id_input": "8",
            "unet_block_id_middle": "",
            "unet_block_id_output": "",
            "unet_attn": "attn1"}
        
        kill_cross_input_8 = kill_self_input_8.copy()
        kill_cross_input_8['unet_attn'] = "attn2"
        
        attention_modifiers_positive = []
        attention_modifiers_fake_negative = []

        if patch_cond: attention_modifiers_positive.append(attn_patch) if not light_patch else attention_modifiers_positive.append(attn_patch_light)
        if mute_self_input_layer_8_cond:  attention_modifiers_positive.append(kill_self_input_8)
        if mute_cross_input_layer_8_cond: attention_modifiers_positive.append(kill_cross_input_8)

        if patch_uncond: attention_modifiers_fake_negative.append(attn_patch) if not light_patch else attention_modifiers_fake_negative.append(attn_patch_light)
        if mute_self_input_layer_8_uncond:  attention_modifiers_fake_negative.append(kill_self_input_8)
        if mute_cross_input_layer_8_uncond: attention_modifiers_fake_negative.append(kill_cross_input_8)

        patch_parameters['attention_modifiers_positive']      = attention_modifiers_positive
        patch_parameters['attention_modifiers_fake_negative'] = attention_modifiers_fake_negative

        if attn_mod_for_positive_operation is not None:
            patch_parameters['attention_modifiers_positive'] = patch_parameters['attention_modifiers_positive'] + attn_mod_for_positive_operation
        if attn_mod_for_negative_operation is not None:
            patch_parameters['attention_modifiers_fake_negative'] = patch_parameters['attention_modifiers_fake_negative'] + attn_mod_for_negative_operation
        if attn_mod_for_global_operation is not None:
            patch_parameters["attention_modifiers_global_enabled"] = True
            patch_parameters['attention_modifiers_global'] = attn_mod_for_global_operation

        patch_parameters["uncond_sigma_end"]      = uncond_sigma_end
        patch_parameters["fake_uncond_sigma_end"] = uncond_sigma_end
        patch_parameters["automatic_cfg"] = "hard" if Auto_CFG else "None"

        if save_as_preset:
            patch_parameters["save_as_preset"] = save_as_preset
            patch_parameters["preset_name"] = preset_name

        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **patch_parameters)[0]
        
        return (m, parameters_as_string, )

```
