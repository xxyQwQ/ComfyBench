---
tags:
- ModelGuidance
---

# Automatic CFG - Advanced
## Documentation
- Class name: `Automatic CFG - Advanced`
- Category: `model_patches/Automatic_CFG`
- Output node: `False`

This node represents an advanced configuration mechanism for dynamically adjusting the computational graph of models, particularly focusing on enhancing or modifying attention mechanisms within neural networks. It provides a sophisticated interface for applying custom modifications to the attention layers, enabling fine-tuned control over model behavior and performance.
## Input types
### Required
- **`model`**
    - The model to be modified, providing the base for the computational graph adjustments.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`automatic_cfg`**
    - Specifies the mode of automatic configuration to apply, allowing for predefined or custom adjustments to the model's computational graph.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`skip_uncond`**
    - A flag to skip unconditional generation steps, optimizing the model's performance for specific tasks.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`fake_uncond_start`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`uncond_sigma_start`**
    - The starting value of sigma for unconditional generation, defining the initial state of noise reduction.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`uncond_sigma_end`**
    - The ending value of sigma for unconditional generation, determining the final state of noise reduction.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lerp_uncond`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`lerp_uncond_strength`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`lerp_uncond_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`lerp_uncond_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`subtract_latent_mean`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`subtract_latent_mean_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`subtract_latent_mean_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`latent_intensity_rescale`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`latent_intensity_rescale_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`latent_intensity_rescale_cfg`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`latent_intensity_rescale_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`latent_intensity_rescale_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`cond_exp`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`cond_exp_normalize`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`cond_exp_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`cond_exp_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`cond_exp_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`cond_exp_value`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`uncond_exp`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`uncond_exp_normalize`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`uncond_exp_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`uncond_exp_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`uncond_exp_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`uncond_exp_value`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`fake_uncond_exp`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`fake_uncond_exp_normalize`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`fake_uncond_exp_method`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`fake_uncond_exp_value`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`fake_uncond_multiplier`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`fake_uncond_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`fake_uncond_sigma_end`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`auto_cfg_topk`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`auto_cfg_ref`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`attention_modifiers_global_enabled`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`disable_cond`**
    - unknown
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `unknown`
- **`disable_cond_sigma_start`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`disable_cond_sigma_end`**
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
- **`eval_string_cond`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`eval_string_uncond`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`eval_string_fake`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`args_filter`**
    - unknown
    - Comfy dtype: `STRING`
    - Python dtype: `unknown`
- **`attention_modifiers_positive`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
- **`attention_modifiers_negative`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
- **`attention_modifiers_fake_negative`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
- **`attention_modifiers_global`**
    - unknown
    - Comfy dtype: `ATTNMOD`
    - Python dtype: `unknown`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with applied configurations, reflecting changes in the computational graph and attention mechanisms.
    - Python dtype: `torch.nn.Module`
- **`string`**
    - Comfy dtype: `STRING`
    - A string representation of the parameters used in the configuration, providing insight into the modifications applied.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class advancedDynamicCFG:
    def __init__(self):
        self.last_cfg_ht_one = 8
        self.previous_cond_pred = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),

                                "automatic_cfg" : (["None", "soft", "hard", "hard_squared", "range"], {"default": "hard"},),

                                "skip_uncond" : ("BOOLEAN", {"default": True}),
                                "fake_uncond_start" : ("BOOLEAN", {"default": False}),
                                "uncond_sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "lerp_uncond" : ("BOOLEAN", {"default": False}),
                                "lerp_uncond_strength":    ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                                "lerp_uncond_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "lerp_uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "subtract_latent_mean" : ("BOOLEAN", {"default": False}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": False}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg": ("FLOAT", {"default": 8,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 3, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "cond_exp": ("BOOLEAN", {"default": False}),
                                "cond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "cond_exp_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_method": (["amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "previous_average", "eval",
                                                     "attention_modifiers_input_using_cond","attention_modifiers_input_using_uncond",
                                                     "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"],),
                                "cond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 100, "step": 0.1, "round": 0.01}),
                                
                                "uncond_exp": ("BOOLEAN", {"default": False}),
                                "uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "uncond_exp_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_method": (["amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "previous_average", "eval",
                                                       "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"],),
                                "uncond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 100, "step": 0.1, "round": 0.01}),

                                "fake_uncond_exp": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_method" : (["cond_pred", "previous_average",
                                                             "amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff",
                                                             "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "eval",
                                                             "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond",
                                                             "attention_modifiers_input_using_cond","attention_modifiers_input_using_uncond"],),
                                "fake_uncond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 1000, "step": 0.1, "round": 0.01}),
                                "fake_uncond_multiplier": ("INT", {"default": 1, "min": -1, "max": 1, "step": 1}),
                                "fake_uncond_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 1,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "auto_cfg_topk":    ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05, "round": 0.01}),
                                "auto_cfg_ref":    ("FLOAT", {"default": 8, "min": 0.0, "max": 100, "step": 0.5, "round": 0.01}),
                                "attention_modifiers_global_enabled": ("BOOLEAN", {"default": False}),
                                "disable_cond": ("BOOLEAN", {"default": False}),
                                "disable_cond_sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "disable_cond_sigma_end":   ("FLOAT", {"default": 0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "save_as_preset": ("BOOLEAN", {"default": False}),
                                "preset_name": ("STRING", {"multiline": False}),
                              },
                              "optional":{
                                  "eval_string_cond": ("STRING", {"multiline": True}),
                                  "eval_string_uncond": ("STRING", {"multiline": True}),
                                  "eval_string_fake": ("STRING", {"multiline": True}),
                                  "args_filter": ("STRING", {"multiline": True, "forceInput": True}),
                                  "attention_modifiers_positive": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_negative": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_fake_negative": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_global": ("ATTNMOD", {"forceInput": True}),
                              }
                              }
    RETURN_TYPES = ("MODEL","STRING",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, automatic_cfg = "None",
              skip_uncond = False, fake_uncond_start = False, uncond_sigma_start = 1000, uncond_sigma_end = 0,
              lerp_uncond = False, lerp_uncond_strength = 1, lerp_uncond_sigma_start = 1000, lerp_uncond_sigma_end = 1,
              subtract_latent_mean     = False,   subtract_latent_mean_sigma_start      = 1000, subtract_latent_mean_sigma_end     = 1,
              latent_intensity_rescale = False,   latent_intensity_rescale_sigma_start  = 1000, latent_intensity_rescale_sigma_end = 1,
              cond_exp = False, cond_exp_sigma_start  = 1000, cond_exp_sigma_end = 1000, cond_exp_method = "amplify", cond_exp_value = 2, cond_exp_normalize = False,
              uncond_exp = False, uncond_exp_sigma_start  = 1000, uncond_exp_sigma_end = 1000, uncond_exp_method = "amplify", uncond_exp_value = 2, uncond_exp_normalize = False,
              fake_uncond_exp = False, fake_uncond_exp_method = "amplify", fake_uncond_exp_value = 2, fake_uncond_exp_normalize = False, fake_uncond_multiplier = 1, fake_uncond_sigma_start = 1000, fake_uncond_sigma_end = 1,
              latent_intensity_rescale_cfg = 8, latent_intensity_rescale_method = "hard",
              ignore_pre_cfg_func = False, args_filter = "", auto_cfg_topk = 0.25, auto_cfg_ref = 8,
              eval_string_cond = "", eval_string_uncond = "", eval_string_fake = "",
              attention_modifiers_global_enabled = False,
              attention_modifiers_positive = [], attention_modifiers_negative = [], attention_modifiers_fake_negative = [], attention_modifiers_global = [],
              disable_cond=False, disable_cond_sigma_start=1000,disable_cond_sigma_end=1000, save_as_preset = False, preset_name = "", **kwargs
              ):
        
        # support_function()
        model_options_copy = deepcopy(model.model_options)
        monkey_patching_comfy_sampling_function()
        if args_filter != "":
            args_filter = args_filter.split(",")
        else:
            args_filter = [k for k, v in locals().items()]
        not_in_filter = ['self','model','args','args_filter','save_as_preset','preset_name','model_options_copy']
        if fake_uncond_exp_method != "eval":
            not_in_filter.append("eval_string")

        if save_as_preset and preset_name != "":
            preset_parameters = {key: value for key, value in locals().items() if key not in not_in_filter}
            with open(os.path.join(json_preset_path, preset_name+".json"), 'w', encoding='utf-8') as f:
                json.dump(preset_parameters, f)
            print(f"Preset saved with the name: {Fore.GREEN}{preset_name}{Fore.RESET}")
            print(f"{Fore.RED}Don't forget to turn the save toggle OFF to not overwrite!{Fore.RESET}")

        args_str = '\n'.join(f'{k}: {v}' for k, v in locals().items() if k not in not_in_filter and k in args_filter)

        sigmin, sigmax = get_sigmin_sigmax(model)

        lerp_start, lerp_end          = lerp_uncond_sigma_start, lerp_uncond_sigma_end
        subtract_start, subtract_end  = subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end
        rescale_start, rescale_end    = latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end
        print(f"Model maximum sigma: {sigmax} / Model minimum sigma: {sigmin}")
        m = model.clone()

        if skip_uncond or disable_cond:
            # set model_options sampler_pre_cfg_function
            m.model_options["sampler_pre_cfg_function"] = make_sampler_pre_cfg_function(uncond_sigma_end if skip_uncond else 0, uncond_sigma_start if skip_uncond else 100000,\
                                                                                        disable_cond_sigma_start if disable_cond else 100000, disable_cond_sigma_end if disable_cond else 100000)
            print(f"Sampling function patched. Uncond enabled from {round(uncond_sigma_start,2)} to {round(uncond_sigma_end,2)}")
        elif not ignore_pre_cfg_func:
            m.model_options.pop("sampler_pre_cfg_function", None)
            uncond_sigma_start, uncond_sigma_end = 1000000, 0

        top_k = auto_cfg_topk
        previous_cond_pred = None
        previous_sigma = None
        def automatic_cfg_function(args):
            nonlocal previous_sigma
            cond_scale = args["cond_scale"]
            input_x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            sigma = args["sigma"][0]
            model_options = args["model_options"]
            if self.previous_cond_pred is None:
                self.previous_cond_pred = cond_pred.clone().detach().to(device=cond_pred.device)
            if previous_sigma is None:
                previous_sigma = sigma.item()
            reference_cfg = auto_cfg_ref if auto_cfg_ref > 0 else cond_scale

            def fake_uncond_step():
                return fake_uncond_start and skip_uncond and (sigma > uncond_sigma_start or sigma < uncond_sigma_end) and sigma <= fake_uncond_sigma_start and sigma >= fake_uncond_sigma_end

            if fake_uncond_step():
                uncond_pred = cond_pred.clone().detach().to(device=cond_pred.device) * fake_uncond_multiplier

            if cond_exp and sigma <= cond_exp_sigma_start and sigma >= cond_exp_sigma_end:
                cond_pred = experimental_functions(cond_pred, cond_exp_method, cond_exp_value, cond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_positive, args, model_options_copy, eval_string_cond)
            if uncond_exp and sigma <= uncond_exp_sigma_start and sigma >= uncond_exp_sigma_end and not fake_uncond_step():
                uncond_pred = experimental_functions(uncond_pred, uncond_exp_method, uncond_exp_value, uncond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_negative, args, model_options_copy, eval_string_uncond)
            if fake_uncond_step() and fake_uncond_exp:
                uncond_pred = experimental_functions(uncond_pred, fake_uncond_exp_method, fake_uncond_exp_value, fake_uncond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_fake_negative, args, model_options_copy, eval_string_fake)
            self.previous_cond_pred = cond_pred.clone().detach().to(device=cond_pred.device)

            if sigma >= sigmax or cond_scale > 1:
                self.last_cfg_ht_one = cond_scale
            target_intensity = self.last_cfg_ht_one / 10

            if ((check_skip(sigma, uncond_sigma_start, uncond_sigma_end) and skip_uncond) and not fake_uncond_step()) or cond_scale == 1:
                return input_x - cond_pred

            if lerp_uncond and not check_skip(sigma, lerp_start, lerp_end) and lerp_uncond_strength != 1:
                uncond_pred_norm = uncond_pred.norm()
                uncond_pred = torch.lerp(cond_pred, uncond_pred, lerp_uncond_strength)
                uncond_pred = uncond_pred * uncond_pred_norm / uncond_pred.norm()
            cond   = input_x - cond_pred
            uncond = input_x - uncond_pred

            if automatic_cfg == "None":
                return uncond + cond_scale * (cond - uncond)

            denoised_tmp = input_x - (uncond + reference_cfg * (cond - uncond))

            for b in range(len(denoised_tmp)):
                denoised_ranges = get_denoised_ranges(denoised_tmp[b], automatic_cfg, top_k)
                for c in range(len(denoised_tmp[b])):
                    fixeds_scale = reference_cfg * target_intensity / denoised_ranges[c]
                    denoised_tmp[b][c] = uncond[b][c] + fixeds_scale * (cond[b][c] - uncond[b][c])

            return denoised_tmp

        def center_mean_latent_post_cfg(args):
            denoised = args["denoised"]
            sigma    = args["sigma"][0]
            if check_skip(sigma, subtract_start, subtract_end):
                return denoised
            denoised = center_latent_mean_values(denoised, False, 1)
            return denoised

        def rescale_post_cfg(args):
            denoised   = args["denoised"]
            sigma      = args["sigma"][0]

            if check_skip(sigma, rescale_start, rescale_end):
                return denoised
            target_intensity = latent_intensity_rescale_cfg / 10
            for b in range(len(denoised)):
                denoised_ranges = get_denoised_ranges(denoised[b], latent_intensity_rescale_method)
                for c in range(len(denoised[b])):
                    scale_correction = target_intensity / denoised_ranges[c]
                    denoised[b][c]   = denoised[b][c] * scale_correction
            return denoised
        
        tmp_model_options = deepcopy(m.model_options)
        if attention_modifiers_global_enabled:
            # print(f"{Fore.GREEN}Sigma timings are ignored for global modifiers.{Fore.RESET}")
            for atm in attention_modifiers_global:
                block_layers = {"input": atm['unet_block_id_input'], "middle": atm['unet_block_id_middle'], "output": atm['unet_block_id_output']}
                for unet_block in block_layers:
                    for unet_block_id in block_layers[unet_block].split(","):
                        if unet_block_id != "":
                            unet_block_id = int(unet_block_id)
                            tmp_model_options = set_model_options_patch_replace(tmp_model_options, attention_modifier(atm['self_attn_mod_eval']).modified_attention, atm['unet_attn'], unet_block, unet_block_id)
            m.model_options = tmp_model_options

        if not ignore_pre_cfg_func:
            m.set_model_sampler_cfg_function(automatic_cfg_function, disable_cfg1_optimization = False)
        if subtract_latent_mean:
            m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)
        if latent_intensity_rescale:
            m.set_model_sampler_post_cfg_function(rescale_post_cfg)
        return (m, args_str, )

```
