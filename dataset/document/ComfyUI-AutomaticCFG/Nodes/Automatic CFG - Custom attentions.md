# Automatic CFG - Custom attentions
## Documentation
- Class name: `Automatic CFG - Custom attentions`
- Category: `model_patches/Automatic_CFG/experimental_attention_modifiers`
- Output node: `False`

This node is designed to concatenate attention modifier parameters from two sources, creating a unified set of parameters for further processing. It serves as a utility within the context of configuring or enhancing model attention mechanisms, particularly in scenarios requiring the combination of distinct attention modification strategies.
## Input types
### Required
- **`model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`Auto_CFG`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`cond_mode`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`uncond_mode`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`cond_diff_multiplier`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`uncond_diff_multiplier`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`uncond_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`save_as_preset`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`preset_name`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
### Optional
- **`attn_mod_for_positive_operation`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
- **`attn_mod_for_negative_operation`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
## Output types
- **`Model`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFGCustomAttentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "Auto_CFG": ("BOOLEAN", {"default": True}),
                                "cond_mode" :   (["replace_by_custom","normal+(normal-custom_cond)*multiplier","normal+(normal-custom_uncond)*multiplier"],),
                                "uncond_mode" : (["replace_by_custom","normal+(normal-custom_cond)*multiplier","normal+(normal-custom_uncond)*multiplier"],),
                                "cond_diff_multiplier":   ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "uncond_diff_multiplier": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":       ("FLOAT", {"default": 1.0, "min":    0.0, "max": 10000, "step": 0.1, "round": 0.01}),
                                "save_as_preset": ("BOOLEAN", {"default": False}),
                                "preset_name":    ("STRING",  {"multiline": False}),
                              },
                              "optional":{
                                  "attn_mod_for_positive_operation": ("ATTNMOD", {"forceInput": True}),
                                  "attn_mod_for_negative_operation": ("ATTNMOD", {"forceInput": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def patch(self, model, Auto_CFG, cond_mode, uncond_mode, cond_diff_multiplier, uncond_diff_multiplier, uncond_sigma_end, save_as_preset, preset_name,
              attn_mod_for_positive_operation = [], attn_mod_for_negative_operation = []):
                
        with open(os.path.join(json_preset_path, "do_not_delete.json"), 'r', encoding='utf-8') as f:
            patch_parameters = json.load(f)
        
        patch_parameters["cond_exp_value"]   = cond_diff_multiplier
        patch_parameters["uncond_exp_value"] = uncond_diff_multiplier

        if cond_mode != "replace_by_custom":
            patch_parameters["disable_cond"] = False
        if cond_mode == "normal+(normal-custom_cond)*multiplier":
            patch_parameters["cond_exp_method"] = "subtract_attention_modifiers_input_using_cond"
        elif cond_mode == "normal+(normal-custom_uncond)*multiplier":
            patch_parameters["cond_exp_method"] = "subtract_attention_modifiers_input_using_uncond"

        if uncond_mode != "replace_by_custom":
            patch_parameters["uncond_sigma_start"] = 1000.0
            patch_parameters["fake_uncond_exp"]    = False
            patch_parameters["uncond_exp"]         = True
        
        if uncond_mode == "normal+(normal-custom_cond)*multiplier":
            patch_parameters["uncond_exp_method"] = "subtract_attention_modifiers_input_using_cond"
        elif uncond_mode == "normal+(normal-custom_uncond)*multiplier":
            patch_parameters["uncond_exp_method"] = "subtract_attention_modifiers_input_using_uncond"

        if cond_mode != "replace_by_custom" and attn_mod_for_positive_operation != []:
            smallest_sigma = min([float(x['sigma_end']) for x in attn_mod_for_positive_operation])
            patch_parameters["disable_cond_sigma_end"] = smallest_sigma
            patch_parameters["cond_exp_sigma_end"]     = smallest_sigma

        if uncond_mode != "replace_by_custom" and attn_mod_for_negative_operation != []:
            smallest_sigma = min([float(x['sigma_end'])   for x in attn_mod_for_negative_operation])
            patch_parameters["uncond_exp_sigma_end"] = smallest_sigma
            patch_parameters["fake_uncond_start"]    = False
        # else:
        #     biggest_sigma = max([float(x['sigma_start']) for x in attn_mod_for_negative_operation])
        #     patch_parameters["fake_uncond_sigma_start"] = biggest_sigma

        patch_parameters["automatic_cfg"] = "hard" if Auto_CFG else "None"
        patch_parameters['attention_modifiers_positive']      = attn_mod_for_positive_operation
        patch_parameters['attention_modifiers_negative']      = attn_mod_for_negative_operation
        patch_parameters['attention_modifiers_fake_negative'] = attn_mod_for_negative_operation
        patch_parameters["uncond_sigma_end"]      = uncond_sigma_end
        patch_parameters["fake_uncond_sigma_end"] = uncond_sigma_end
        patch_parameters["save_as_preset"] = save_as_preset
        patch_parameters["preset_name"]    = preset_name
        
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **patch_parameters)[0]
        
        return (m, )

```
