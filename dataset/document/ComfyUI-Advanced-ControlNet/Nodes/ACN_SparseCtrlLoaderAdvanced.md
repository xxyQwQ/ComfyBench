---
tags:
- ControlNet
- ControlNetLoader
---

# Load SparseCtrl Model 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_SparseCtrlLoaderAdvanced`
- Category: `Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl`
- Output node: `False`

This node is designed for loading and initializing the Sparse ControlNet models, facilitating the integration of sparse control mechanisms into the advanced control network architecture. It abstracts the complexities of model loading, ensuring that the Sparse ControlNet models are properly set up for subsequent operations.
## Input types
### Required
- **`sparsectrl_name`**
    - Specifies the name of the Sparse ControlNet model to be loaded, crucial for identifying the correct model file.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_motion`**
    - A boolean flag that determines whether motion-based control mechanisms should be applied, influencing the dynamic aspects of the control.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`motion_strength`**
    - Defines the intensity of the motion effect, adjusting how strongly motion influences the control mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`motion_scale`**
    - Sets the scale of motion effects, modifying the extent to which motion is factored into the control process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`sparse_method`**
    - Selects the method for sparse control, impacting the strategy for sparse adjustments within the network.
    - Comfy dtype: `SPARSE_METHOD`
    - Python dtype: `str`
- **`tk_optional`**
    - Optional timestep keyframe input for more granular control over animation or motion effects.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `str`
- **`context_aware`**
    - Enables context-aware adjustments, allowing the model to adapt its control strategy based on contextual information.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`sparse_hint_mult`**
    - Multiplier for hints in sparse control, affecting how hint-based adjustments are scaled.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sparse_nonhint_mult`**
    - Multiplier for non-hint adjustments, influencing the strength of control without explicit hints.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`sparse_mask_mult`**
    - Multiplier for mask-based adjustments in sparse control, determining the influence of mask inputs on the control strategy.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`control_net`**
    - Comfy dtype: `CONTROL_NET`
    - The loaded Sparse ControlNet model, ready for integration and use within the advanced control network.
    - Python dtype: `object`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)



## Source code
```python
class SparseCtrlLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sparsectrl_name": (folder_paths.get_filename_list("controlnet"), ),
                "use_motion": ("BOOLEAN", {"default": True}, ),
                "motion_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            },
            "optional": {
                "sparse_method": ("SPARSE_METHOD", ),
                "tk_optional": ("TIMESTEP_KEYFRAME", ),
                "context_aware": (SparseContextAware.LIST, ),
                "sparse_hint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_nonhint_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "sparse_mask_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET", )
    FUNCTION = "load_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝/SparseCtrl"

    def load_controlnet(self, sparsectrl_name: str, use_motion: bool, motion_strength: float, motion_scale: float, sparse_method: SparseMethod=SparseSpreadMethod(), tk_optional: TimestepKeyframeGroup=None,
                        context_aware=SparseContextAware.NEAREST_HINT, sparse_hint_mult=1.0, sparse_nonhint_mult=1.0, sparse_mask_mult=1.0):
        sparsectrl_path = folder_paths.get_full_path("controlnet", sparsectrl_name)
        sparse_settings = SparseSettings(sparse_method=sparse_method, use_motion=use_motion, motion_strength=motion_strength, motion_scale=motion_scale,
                                         context_aware=context_aware,
                                         sparse_mask_mult=sparse_mask_mult, sparse_hint_mult=sparse_hint_mult, sparse_nonhint_mult=sparse_nonhint_mult)
        sparsectrl = load_sparsectrl(sparsectrl_path, timestep_keyframe=tk_optional, sparse_settings=sparse_settings)
        return (sparsectrl,)

```
