---
tags:
- ModelMerge
---

# multiModelMerge
## Documentation
- Class name: `ttN multiModelMerge`
- Category: `🌏 tinyterra`
- Output node: `False`

The `ttN_multiModelMerge` node specializes in combining multiple models or CLIPs through various interpolation methods, enabling the creation of hybrid models that leverage the strengths of each input model. It supports a range of interpolation strategies, such as weighted sum and addition of differences, to fine-tune the merging process according to specific requirements.
## Input types
### Required
- **`ckpt_A_name`**
    - Specifies the checkpoint name for model A, used as an input for model merging.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_A_name`**
    - Specifies the configuration name for model A, used alongside the checkpoint name to load the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`ckpt_B_name`**
    - Specifies the checkpoint name for model B, used as an input for model merging.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_B_name`**
    - Specifies the configuration name for model B, used alongside the checkpoint name to load the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`ckpt_C_name`**
    - Specifies the checkpoint name for model C, used as an input for model merging.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`config_C_name`**
    - Specifies the configuration name for model C, used alongside the checkpoint name to load the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model_interpolation`**
    - Defines the interpolation method to be used for merging models.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`model_multiplier`**
    - The multiplier factor used in the interpolation of models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`clip_interpolation`**
    - Defines the interpolation method to be used for merging CLIP models.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`clip_multiplier`**
    - The multiplier factor used in the interpolation of CLIP models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`model_A_override`**
    - Overrides the model A with a specified model, bypassing the need for a checkpoint.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`model_B_override`**
    - Overrides the model B with a specified model, bypassing the need for a checkpoint.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`model_C_override`**
    - Overrides the model C with a specified model, bypassing the need for a checkpoint.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`clip_A_override`**
    - Overrides the CLIP model A with a specified CLIP, bypassing the need for a checkpoint.
    - Comfy dtype: `CLIP`
    - Python dtype: `comfy.model_base.CLIP`
- **`clip_B_override`**
    - Overrides the CLIP model B with a specified CLIP, bypassing the need for a checkpoint.
    - Comfy dtype: `CLIP`
    - Python dtype: `comfy.model_base.CLIP`
- **`clip_C_override`**
    - Overrides the CLIP model C with a specified CLIP, bypassing the need for a checkpoint.
    - Comfy dtype: `CLIP`
    - Python dtype: `comfy.model_base.CLIP`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The resulting merged model, which is a hybrid of the input models, incorporating features from both according to the chosen interpolation method.
    - Python dtype: `comfy.model_base.Model`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The resulting merged CLIP model, combining the features of both input CLIP models based on the specified interpolation strategy.
    - Python dtype: `comfy.model_base.CLIP`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_multiModelMerge:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_A_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "config_A_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                    "ckpt_B_name": (["None",] + folder_paths.get_filename_list("checkpoints"), ),
                    "config_B_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                    "ckpt_C_name": (["None",] + folder_paths.get_filename_list("checkpoints"), ),
                    "config_C_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),

                    "model_interpolation": (MODEL_INTERPOLATIONS,),
                    "model_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    
                    "clip_interpolation": (CLIP_INTERPOLATIONS,),
                    "clip_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "model_A_override": ("MODEL",),
                    "model_B_override": ("MODEL",),
                    "model_C_override": ("MODEL",),
                    "clip_A_override": ("CLIP",),
                    "clip_B_override": ("CLIP",),
                    "clip_C_override": ("CLIP",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "ttNnodeVersion": ttN_multiModelMerge.version, "my_unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("model", "clip",)
    FUNCTION = "mergificate"

    CATEGORY = "🌏 tinyterra"

    def mergificate(self, ckpt_A_name, config_A_name, ckpt_B_name, config_B_name, ckpt_C_name, config_C_name,
                model_interpolation, model_multiplier, clip_interpolation, clip_multiplier,
                model_A_override=None, model_B_override=None, model_C_override=None,
                clip_A_override=None, clip_B_override=None, clip_C_override=None,
                prompt=None, extra_pnginfo=None, my_unique_id=None):
        
        def required_assets(model_interpolation, clip_interpolation):
            required = set(["model_A"])
            
            if clip_interpolation in [A_ONLY, B_ONLY, C_ONLY]:
                required.add(f"clip_{clip_interpolation[0]}")
            elif clip_interpolation in [WEIGHTED_SUM, ADD_DIFFERENCE]:
                required.update([f"clip_{letter}" for letter in ABC if letter in clip_interpolation])
            elif clip_interpolation == FOLLOW:
                required.add("clip_A")
            
            if model_interpolation in [WEIGHTED_SUM, ADD_DIFFERENCE]:
                letters = [letter for letter in ABC if letter in model_interpolation]
                required.update([f"model_{letter}" for letter in letters])
                if clip_interpolation == FOLLOW:
                    required.update([f"clip_{letter}" for letter in letters])
            
            return sorted(list(required))

        def _collect_letter(letter, required_list, model_override, clip_override, ckpt_name, config_name = None):
            model, clip, loaded_clip = None, None, None
            config_name = config_name
            
            if f'model_{letter}' in required_list:
                if model_override not in [None, "None"]:
                    model = model_override
                else:
                    if ckpt_name not in [None, "None"]:
                        model, loaded_clip, _ = loader.load_checkpoint(ckpt_name, config_name)
                    else:
                        e = f"Checkpoint name or model override not provided for model_{letter}.\nUnable to merge models using the following interpolation: {model_interpolation}"
                        ttNl(e).t(f'multiModelMerge [{my_unique_id}]').error().p().interrupt(e)
            
            if f'clip_{letter}' in required_list:
                if clip_override is not None:
                    clip = clip_override
                elif loaded_clip is not None:
                    clip = loaded_clip
                elif ckpt_name not in [None, "None"]:
                    _, clip, _ = loader.load_checkpoint(ckpt_name, config_name)
                else:
                    e = f"Checkpoint name or clip override not provided for clip_{letter}.\nUnable to merge clips using the following interpolation: {clip_interpolation}"
                    ttNl(e).t(f'multiModelMerge [{my_unique_id}]').error().p().interrupt(e)
            
            return model, clip

        def merge(base_model, base_strength, patch_model, patch_strength):
            m = base_model.clone()
            kp = patch_model.get_key_patches("diffusion_model.")
            for k in kp:
                m.add_patches({k: kp[k]}, patch_strength, base_strength)
            return m
        
        def clip_merge(base_clip, base_strength, patch_clip, patch_strength):
            m = base_clip.clone()
            kp = patch_clip.get_key_patches()
            for k in kp:
                if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                    continue
                m.add_patches({k: kp[k]}, patch_strength, base_strength)
            return m

        def _add_assets(a1, a2, is_clip=False, multiplier=1.0, weighted=False):
            if is_clip:
                if weighted:
                    return clip_merge(a1, (1.0 - multiplier), a2, multiplier)
                else:
                    return clip_merge(a1, 1.0, a2, multiplier)
            else:
                if weighted:
                    return merge(a1, (1.0 - multiplier), a2, multiplier)
                else:
                    return merge(a1, 1.0, a2, multiplier)
        
        def _subtract_assets(a1, a2, is_clip=False, multiplier=1.0):
            if is_clip:
                    return clip_merge(a1, 1.0, a2, -multiplier)
            else:
                    return merge(a1, 1.0, a2, -multiplier)
        
        required_list = required_assets(model_interpolation, clip_interpolation)
        model_A, clip_A = _collect_letter("A", required_list, model_A_override, clip_A_override, ckpt_A_name, config_A_name)
        model_B, clip_B = _collect_letter("B", required_list, model_B_override, clip_B_override, ckpt_B_name, config_B_name)
        model_C, clip_C = _collect_letter("C", required_list, model_C_override, clip_C_override, ckpt_C_name, config_C_name)
        
        if (model_interpolation == A_ONLY):
            model = model_A
        if (model_interpolation == WEIGHTED_SUM):
            model = _add_assets(model_A, model_B, False, model_multiplier, True)
        if (model_interpolation == ADD_DIFFERENCE):
            model = _add_assets(model_A, _subtract_assets(model_B, model_C), False, model_multiplier)
        
        if (clip_interpolation == FOLLOW):
            clip_interpolation = model_interpolation
        if (clip_interpolation == A_ONLY):
            clip = clip_A
        if (clip_interpolation == B_ONLY):
            clip = clip_B
        if (clip_interpolation == C_ONLY):
            clip = clip_C
        if (clip_interpolation == WEIGHTED_SUM):
            clip = _add_assets(clip_A, clip_B, True, clip_multiplier, True)
        if (clip_interpolation == ADD_DIFFERENCE):
            clip = _add_assets(clip_A, _subtract_assets(clip_B, clip_C, True), True, clip_multiplier)

        return (model, clip)

```
