---
tags:
- Conditioning
---

# LayerUtility: SD3 Negative Conditioning
## Documentation
- Class name: `LayerUtility: SD3NegativeConditioning`
- Category: `😺dzNodes/LayerUtility/SystemIO`
- Output node: `False`

This node applies negative conditioning to the input conditioning data for SD3 models, aiming to modify and control the generation process by zeroing out specific parts of the conditioning and adjusting the range of effect. It's designed to enhance or suppress certain aspects of the generated content based on the negative conditioning applied.
## Input types
### Required
- **`conditioning`**
    - The input conditioning data to be modified. It's crucial for defining the aspects of the generated content that need enhancement or suppression.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
- **`zero_out_start`**
    - Defines the starting point (as a percentage of the total length) from which the negative conditioning effect begins, allowing for precise control over the area of impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The modified conditioning data after applying negative conditioning, ready to be used in the generation process.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SD3NegativeConditioning:

    @classmethod
    def INPUT_TYPES(self):
        return {"required": {"conditioning": ("CONDITIONING", ),
                             "zero_out_start": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.001}),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = 'sd3_negative_conditioning'
    CATEGORY = '😺dzNodes/LayerUtility/SystemIO'

    def sd3_negative_conditioning(self, conditioning, zero_out_start):
        def zero_out(conditioning):
            c = []
            for t in conditioning:
                d = t[1].copy()
                if "pooled_output" in d:
                    d["pooled_output"] = torch.zeros_like(d["pooled_output"])
                n = [torch.zeros_like(t[0]), d]
                c.append(n)
            return c

        def set_range(conditioning, start, end):
            c = node_helpers.conditioning_set_values(conditioning, {"start_percent": start,
                                                                    "end_percent": end})
            return c

        zero_out_c = zero_out(conditioning)
        c_1 = set_range(zero_out_c, zero_out_start, 1.0)
        c_2 = set_range(conditioning, 0.0, zero_out_start)
        return (c_1 + c_2,)

```
