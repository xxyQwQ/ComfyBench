---
tags:
- AnimateDiff
- Animation
---

# 🚫[DEPR] Motion Model Settings (Adv. Attn) 🎭🅐🅓①
## Documentation
- Class name: `ADE_AnimateDiffModelSettingsAdvancedAttnStrengths`
- Category: ``
- Output node: `False`

This node is designed to fine-tune the attention strengths within the AnimateDiff model, allowing for advanced customization of the model's attention mechanisms. It provides the capability to adjust the multipliers for various components of the attention mechanism, including query, key, value, output weights, and biases, thereby offering a granular control over the model's focus and processing of input features.
## Input types
### Required
- **`pe_strength`**
    - Specifies the multiplier for positional encoding adjustments, influencing how the model perceives the position of elements within the input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_strength`**
    - Defines the overall multiplier for the attention mechanism, affecting the model's ability to focus on relevant parts of the input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_q_strength`**
    - Adjusts the multiplier for the query component of the attention mechanism, impacting how queries are generated from the input.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_k_strength`**
    - Sets the multiplier for the key component, influencing the matching process between queries and keys.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_v_strength`**
    - Modifies the multiplier for the value component, affecting the information retrieved based on query-key matching.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_out_weight_strength`**
    - Controls the multiplier for the attention output weights, adjusting the influence of attention outputs on the final result.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`attn_out_bias_strength`**
    - Determines the multiplier for the attention output biases, fine-tuning the bias applied to the attention outputs.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`other_strength`**
    - Specifies the multiplier for other model components not explicitly covered by the other parameters, offering a catch-all adjustment mechanism.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`motion_pe_stretch`**
    - Specifies the stretch factor for positional encoding in motion, adjusting how motion influences positional encoding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cap_initial_pe_length`**
    - Caps the initial length of positional encoding, limiting the extent of positional encoding adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`interpolate_pe_to_length`**
    - Determines the length to which positional encoding is interpolated, affecting the granularity of positional adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`initial_pe_idx_offset`**
    - Sets the initial index offset for positional encoding, adjusting the starting point of positional encoding.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`final_pe_idx_offset`**
    - Defines the final index offset for positional encoding, adjusting the endpoint of positional encoding adjustments.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`mask_motion_scale`**
    - Specifies a tensor for scaling the attention mask based on motion, allowing for dynamic adjustment of attention based on motion.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`min_motion_scale`**
    - Sets the minimum scale for motion-based attention adjustments, ensuring a baseline level of motion influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`max_motion_scale`**
    - Defines the maximum scale for motion-based attention adjustments, capping the influence of motion on attention.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`ad_settings`**
    - Comfy dtype: `AD_SETTINGS`
    - Returns an updated AnimateDiffSettings object that incorporates the specified attention and motion-related adjustments, enabling subsequent customization steps.
    - Python dtype: `AnimateDiffSettings`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class AnimateDiffModelSettingsAdvancedAttnStrengths:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pe_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_q_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_k_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_v_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_weight_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "attn_out_bias_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "other_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.0001}),
                "motion_pe_stretch": ("INT", {"default": 0, "min": 0, "step": 1}),
                "cap_initial_pe_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "interpolate_pe_to_length": ("INT", {"default": 0, "min": 0, "step": 1}),
                "initial_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
                "final_pe_idx_offset": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "optional": {
                "mask_motion_scale": ("MASK",),
                "min_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "max_motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "optional": {"deprecation_warning": ("ADEWARN", {"text": "Deprecated"})},
            }
        }
    
    RETURN_TYPES = ("AD_SETTINGS",)
    CATEGORY = ""  #"Animate Diff 🎭🅐🅓/① Gen1 nodes ①/motion settings/experimental"
    FUNCTION = "get_motion_model_settings"

    def get_motion_model_settings(self, pe_strength: float, attn_strength: float,
                                  attn_q_strength: float,
                                  attn_k_strength: float,
                                  attn_v_strength: float,
                                  attn_out_weight_strength: float,
                                  attn_out_bias_strength: float,
                                  other_strength: float,
                                  motion_pe_stretch: int,
                                  cap_initial_pe_length: int, interpolate_pe_to_length: int,
                                  initial_pe_idx_offset: int, final_pe_idx_offset: int,
                                  mask_motion_scale: torch.Tensor=None, min_motion_scale: float=1.0, max_motion_scale: float=1.0):
        adjust_pe = AdjustGroup(AdjustPE(motion_pe_stretch=motion_pe_stretch,
                             cap_initial_pe_length=cap_initial_pe_length, interpolate_pe_to_length=interpolate_pe_to_length,
                             initial_pe_idx_offset=initial_pe_idx_offset, final_pe_idx_offset=final_pe_idx_offset))
        adjust_weight = AdjustGroup(AdjustWeight(
            pe_MULT=pe_strength,
            attn_MULT=attn_strength,
            attn_q_MULT=attn_q_strength,
            attn_k_MULT=attn_k_strength,
            attn_v_MULT=attn_v_strength,
            attn_out_weight_MULT=attn_out_weight_strength,
            attn_out_bias_MULT=attn_out_bias_strength,
            other_MULT=other_strength,
        ))
        motion_model_settings = AnimateDiffSettings(
            adjust_pe=adjust_pe,
            adjust_weight=adjust_weight,
            mask_attn_scale=mask_motion_scale,
            mask_attn_scale_min=min_motion_scale,
            mask_attn_scale_max=max_motion_scale,
        )

        return (motion_model_settings,)

```
