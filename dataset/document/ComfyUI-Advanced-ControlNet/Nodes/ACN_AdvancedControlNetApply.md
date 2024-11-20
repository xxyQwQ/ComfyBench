---
tags:
- ControlNet
---

# Apply Advanced ControlNet 🛂🅐🅒🅝
## Documentation
- Class name: `ACN_AdvancedControlNetApply`
- Category: `Adv-ControlNet 🛂🅐🅒🅝`
- Output node: `False`

This node applies advanced control net transformations to conditioning data, enhancing image generation with more nuanced control and customization options. It leverages advanced features of control nets to modify and fine-tune the conditioning process, allowing for sophisticated manipulation of image attributes.
## Input types
### Required
- **`positive`**
    - Positive conditioning data that will be enhanced through the application of control net transformations, contributing to the generation of desired image attributes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`negative`**
    - Negative conditioning data that will be modified by control net transformations to avoid certain image attributes, ensuring the generated image aligns with specified preferences.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[str, Dict]]`
- **`control_net`**
    - The advanced control net model used to apply transformations to the conditioning data, enabling the customization and fine-tuning of image attributes.
    - Comfy dtype: `CONTROL_NET`
    - Python dtype: `ControlNetAdvanced`
- **`image`**
    - The input image to which the control net transformations are applied, serving as a canvas for the enhanced conditioning effects.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`strength`**
    - A scalar value determining the intensity of the control net transformations applied to the conditioning data, allowing for fine-tuning of the effect's strength.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_percent`**
    - Defines the starting percentage of the effect's application, allowing for gradual introduction of the control net transformations over the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_percent`**
    - Specifies the ending percentage for the application of control net transformations, enabling a controlled and gradual effect on the image.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`mask_optional`**
    - An optional mask that can be applied to selectively target areas for control net transformations, enhancing precision in image manipulation.
    - Comfy dtype: `MASK`
    - Python dtype: `torch.Tensor`
- **`timestep_kf`**
    - Optional keyframe for timestep control, allowing for dynamic adjustment of transformation application over time.
    - Comfy dtype: `TIMESTEP_KEYFRAME`
    - Python dtype: `TimestepKeyframe`
- **`latent_kf_override`**
    - Optional override for latent keyframes, providing advanced control over the conditioning process through specific latent manipulations.
    - Comfy dtype: `LATENT_KEYFRAME`
    - Python dtype: `LatentKeyframe`
- **`weights_override`**
    - Optional weights override for the control net, offering further customization of the transformation effects based on specific weight adjustments.
    - Comfy dtype: `CONTROL_NET_WEIGHTS`
    - Python dtype: `ControlNetWeights`
- **`model_optional`**
    - An optional model parameter that can be used in conjunction with the control net for enhanced conditioning effects.
    - Comfy dtype: `MODEL`
    - Python dtype: `Model`
- **`vae_optional`**
    - An optional VAE parameter that can be utilized for additional conditioning manipulations, enriching the image generation process.
    - Comfy dtype: `VAE`
    - Python dtype: `VAE`
- **`autosize`**
    - Optional autosizing parameter that automatically adjusts the image size for optimal processing, enhancing the efficiency of transformation application.
    - Comfy dtype: `ACNAUTOSIZE`
    - Python dtype: `ACNAUTOSIZE`
## Output types
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The enhanced positive conditioning data after the application of advanced control net transformations.
    - Python dtype: `List[Tuple[str, Dict]]`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The modified negative conditioning data following the application of control net transformations, tailored to avoid specific attributes.
    - Python dtype: `List[Tuple[str, Dict]]`
- **`model_opt`**
    - Comfy dtype: `MODEL`
    - An optional output model that has been adjusted through the control net application process, reflecting changes in conditioning effects.
    - Python dtype: `Model`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [ACN_AdvancedControlNetApply](../../ComfyUI-Advanced-ControlNet/Nodes/ACN_AdvancedControlNetApply.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [KSampler (Efficient)](../../efficiency-nodes-comfyui/Nodes/KSampler (Efficient).md)
    - [ConditioningSetMask](../../Comfy/Nodes/ConditioningSetMask.md)



## Source code
```python
class AdvancedControlNetApply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "timestep_kf": ("TIMESTEP_KEYFRAME", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                "weights_override": ("CONTROL_NET_WEIGHTS", ),
                "model_optional": ("MODEL",),
                "vae_optional": ("VAE",),
                "autosize": ("ACNAUTOSIZE", {"padding": 40}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","MODEL",)
    RETURN_NAMES = ("positive", "negative", "model_opt")
    FUNCTION = "apply_controlnet"

    CATEGORY = "Adv-ControlNet 🛂🅐🅒🅝"

    def apply_controlnet(self, positive, negative, control_net, image, strength, start_percent, end_percent,
                         mask_optional: Tensor=None, model_optional: ModelPatcher=None, vae_optional=None,
                         timestep_kf: TimestepKeyframeGroup=None, latent_kf_override: LatentKeyframeGroup=None,
                         weights_override: ControlWeights=None):
        if strength == 0:
            return (positive, negative, model_optional)
        if model_optional:
            model_optional = model_optional.clone()

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    # copy, convert to advanced if needed, and set cond
                    c_net = convert_to_advanced(control_net.copy()).set_cond_hint(control_hint, strength, (start_percent, end_percent), vae_optional)
                    if is_advanced_controlnet(c_net):
                        # disarm node check
                        c_net.disarm()
                        # if model required, verify model is passed in, and if so patch it
                        if c_net.require_model:
                            if not model_optional:
                                raise Exception(f"Type '{type(c_net).__name__}' requires model_optional input, but got None.")
                            c_net.patch_model(model=model_optional)
                        # if vae required, verify vae is passed in
                        if c_net.require_vae:
                            # if controlnet can accept preprocced condhint latents and is the case, ignore vae requirement
                            if c_net.allow_condhint_latents and isinstance(control_hint, AbstractPreprocWrapper):
                                pass
                            elif not vae_optional:
                                # make sure SD3 ControlNet will get a special message instead of generic type mention
                                if is_sd3_advanced_controlnet:
                                    raise Exception(f"SD3 ControlNet requires vae_optional input, but got None.")
                                else:
                                    raise Exception(f"Type '{type(c_net).__name__}' requires vae_optional input, but got None.")
                        # apply optional parameters and overrides, if provided
                        if timestep_kf is not None:
                            c_net.set_timestep_keyframes(timestep_kf)
                        if latent_kf_override is not None:
                            c_net.latent_keyframe_override = latent_kf_override
                        if weights_override is not None:
                            c_net.weights_override = weights_override
                        # verify weights are compatible
                        c_net.verify_all_weights()
                        # set cond hint mask
                        if mask_optional is not None:
                            mask_optional = mask_optional.clone()
                            # if not in the form of a batch, make it so
                            if len(mask_optional.shape) < 3:
                                mask_optional = mask_optional.unsqueeze(0)
                            c_net.set_cond_hint_mask(mask_optional)
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], model_optional)

```
