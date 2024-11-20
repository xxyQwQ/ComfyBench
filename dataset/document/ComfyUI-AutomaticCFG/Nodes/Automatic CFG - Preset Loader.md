---
tags:
- Animation
---

# Automatic CFG - Preset Loader
## Documentation
- Class name: `Automatic CFG - Preset Loader`
- Category: `model_patches/Automatic_CFG`
- Output node: `False`

This node is designed to dynamically adjust the configuration of generative models by applying predefined or custom presets. It enhances model performance and flexibility by allowing users to modify model behavior based on specific presets, which can include adjustments to conditional and unconditional generation parameters, attention mechanisms, and other model-specific settings.
## Input types
### Required
- **`model`**
    - The generative model to which the preset configurations will be applied. This parameter is crucial as it determines the base model that will be modified according to the selected preset.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`preset`**
    - The name of the preset to be applied to the model. Presets define a set of parameters and modifications that tailor the model's behavior for specific tasks or performance characteristics.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`uncond_sigma_end`**
    - Specifies the end value for the unconditional sigma parameter, allowing for fine-tuning of the model's unconditional generation capabilities.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`use_uncond_sigma_end_from_preset`**
    - A boolean flag indicating whether to use the unconditional sigma end value specified in the preset or the value provided directly to the node.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`automatic_cfg`**
    - Determines the mode of automatic configuration to be applied, which can range from preset-defined settings to custom configurations aimed at optimizing model performance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`join_global_parameters`**
    - Optional parameters that can be globally applied to the model, enhancing its capabilities or modifying its behavior in conjunction with the selected preset.
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `Dict[str, Any]`
## Output types
- **`Model`**
    - Comfy dtype: `MODEL`
    - unknown
    - Python dtype: `unknown`
- **`Preset name`**
    - Comfy dtype: `STRING`
    - unknown
    - Python dtype: `unknown`
- **`Parameters as string`**
    - Comfy dtype: `STRING`
    - A detailed string representation of the applied preset parameters, providing insight into the modifications made to the model.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class presetLoader:
    @classmethod
    def INPUT_TYPES(s):
        presets_files = [pj.replace(".json","") for pj in os.listdir(json_preset_path) if ".json" in pj and pj not in ["Experimental_temperature.json","do_not_delete.json"]]
        presets_files = sorted(presets_files, key=str.lower)
        return {"required": {
                                "model": ("MODEL",),
                                "preset" : (presets_files, {"default": "Excellent_attention"}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "use_uncond_sigma_end_from_preset" : ("BOOLEAN", {"default": True}),
                                "automatic_cfg" : (["From preset","None", "soft", "hard", "hard_squared", "range"],),
                              },
                              "optional":{
                                  "join_global_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    RETURN_TYPES = ("MODEL", "STRING", "STRING",)
    RETURN_NAMES = ("Model", "Preset name", "Parameters as string",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, preset, uncond_sigma_end, use_uncond_sigma_end_from_preset, automatic_cfg, join_global_parameters=None):
        with open(os.path.join(json_preset_path, preset+".json"), 'r', encoding='utf-8') as f:
            preset_args = json.load(f)
        if not use_uncond_sigma_end_from_preset:
            preset_args["uncond_sigma_end"] = uncond_sigma_end
            preset_args["fake_uncond_sigma_end"] = uncond_sigma_end
            preset_args["fake_uncond_exp_sigma_end"] = uncond_sigma_end
            preset_args["uncond_exp_sigma_end"] = uncond_sigma_end            
        
        if join_global_parameters is not None:
            preset_args["attention_modifiers_global"] = preset_args["attention_modifiers_global"] + join_global_parameters
            preset_args["attention_modifiers_global_enabled"] = True

        if automatic_cfg != "From preset":
            preset_args["automatic_cfg"] = automatic_cfg
        
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **preset_args)[0]
        info_string = ",\n".join([f"\"{k}\": {v}" for k,v in preset_args.items() if v != ""])
        print(f"Preset {Fore.GREEN}{preset}{Fore.RESET} loaded successfully!")
        return (m, preset, info_string,)

```
