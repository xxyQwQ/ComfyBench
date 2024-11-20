---
tags:
- ModelGuidance
---

# Automatic CFG - Negative
## Documentation
- Class name: `Automatic CFG - Negative`
- Category: `model_patches/Automatic_CFG/presets`
- Output node: `False`

This node is designed to modify the behavior of generative models by applying conditional fine-grained control, specifically targeting the enhancement or suppression of certain features or themes within the generated content. It leverages advanced dynamic configuration techniques to adjust the model's output based on specified negative conditions, aiming to refine the generation process and output quality.
## Input types
### Required
- **`model`**
    - The generative model to be modified. This parameter is crucial as it determines the base model on which the conditional fine-grained control will be applied, affecting the overall generation process and output.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`boost`**
    - A boolean flag indicating whether to skip unconditional generation steps, thereby focusing the model's generation capabilities more directly on the specified conditions. This can lead to more pronounced effects of the negative conditions on the generated content.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`negative_strength`**
    - A floating-point value specifying the strength of the negative conditioning. This affects how strongly the specified negative conditions influence the model's output, allowing for fine-tuned control over the suppression of undesired features or themes.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified generative model, now capable of applying conditional fine-grained control based on the specified negative conditions. This model is expected to produce content that aligns more closely with the desired output characteristics by suppressing undesired features or themes.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class simpleDynamicCFGlerpUncond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "boost" : ("BOOLEAN", {"default": True}),
                                "negative_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, boost, negative_strength):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model,
                         automatic_cfg="hard", skip_uncond=boost,
                         uncond_sigma_start = 15, uncond_sigma_end = 1,
                         lerp_uncond=negative_strength != 1, lerp_uncond_strength=negative_strength,
                         lerp_uncond_sigma_start = 15, lerp_uncond_sigma_end = 1
                         )[0]
        return (m, )

```
