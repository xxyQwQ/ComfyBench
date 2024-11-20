---
tags:
- ModelGuidance
---

# 🔧 SD3 Negative Conditioning
## Documentation
- Class name: `SD3NegativeConditioning+`
- Category: `essentials/conditioning`
- Output node: `False`

This node specializes in modifying the conditioning input for generative models by applying a negative conditioning technique. It zeroes out the initial conditioning and then selectively applies conditioning within a specified timestep range, effectively allowing for more controlled and nuanced generation outcomes.
## Input types
### Required
- **`conditioning`**
    - The conditioning input represents the initial set of conditions or parameters that guide the generative model's output. Modifying this input allows for targeted adjustments to the generation process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `torch.Tensor`
- **`end`**
    - Specifies the end of the timestep range within which conditioning is applied. A value of 0 indicates no conditioning, while values greater than 0 adjust the extent of conditioning applied, influencing the generative model's output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning output, which has been adjusted within the specified timestep range to influence the generative model's behavior.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class SD3NegativeConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "end": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001 }),
        }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials/conditioning"

    def execute(self, conditioning, end):
        zero_c = ConditioningZeroOut().zero_out(conditioning)[0]

        if end == 0:
            return (zero_c, )

        c = ConditioningSetTimestepRange().set_range(conditioning, 0, end)[0]
        zero_c = ConditioningSetTimestepRange().set_range(zero_c, end, 1.0)[0]
        c = ConditioningCombine().combine(zero_c, c)[0]

        return (c, )

```
