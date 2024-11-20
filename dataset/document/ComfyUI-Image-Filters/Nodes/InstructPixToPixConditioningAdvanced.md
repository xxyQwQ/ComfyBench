---
tags:
- Conditioning
---

# InstructPixToPixConditioningAdvanced
## Documentation
- Class name: `InstructPixToPixConditioningAdvanced`
- Category: `conditioning/instructpix2pix`
- Output node: `False`

This node is designed for advanced conditioning in Pix2Pix models, enabling the manipulation and transformation of image data through a sophisticated encoding process. It facilitates the integration of additional conditioning inputs and scales, allowing for more nuanced control over the generation process.
## Input types
### Required
- **`positive`**
    - Specifies the positive conditioning elements, influencing the encoding process to favor certain outcomes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list`
- **`negative`**
    - Specifies the negative conditioning elements, influencing the encoding process to avoid certain outcomes.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `list`
- **`new`**
    - Represents the new input data to be encoded, providing a basis for the generation.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`new_scale`**
    - A scaling factor for the new input data, adjusting its influence in the encoding process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`original`**
    - Represents the original input data, serving as a reference or baseline in the encoding process.
    - Comfy dtype: `LATENT`
    - Python dtype: `dict`
- **`original_scale`**
    - A scaling factor for the original input data, adjusting its influence in the encoding process.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`cond1`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the modified positive conditioning elements after encoding.
    - Python dtype: `list`
- **`cond2`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the modified negative conditioning elements after encoding.
    - Python dtype: `list`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the unaltered negative conditioning elements, maintaining their original state post-encoding.
    - Python dtype: `list`
- **`latent`**
    - Comfy dtype: `LATENT`
    - Generates a new latent representation based on the encoded inputs and scales.
    - Python dtype: `dict`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class InstructPixToPixConditioningAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "new": ("LATENT", ),
                             "new_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                             "original": ("LATENT", ),
                             "original_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("cond1", "cond2", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/instructpix2pix"

    def encode(self, positive, negative, new, new_scale, original, original_scale):
        new_shape, orig_shape = new["samples"].shape, original["samples"].shape
        if new_shape != orig_shape:
            raise Exception(f"Latent shape mismatch: {new_shape} and {orig_shape}")
        
        out_latent = {}
        out_latent["samples"] = new["samples"] * new_scale
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = original["samples"] * original_scale
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], negative, out_latent)

```
