---
tags:
- RegionalPrompt
---

# Regional Conditioning By Color Mask (Inspire)
## Documentation
- Class name: `RegionalConditioningColorMask __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node specializes in applying regional conditioning to an image based on a specified color mask. It combines textual prompts with visual cues from a color mask to generate conditioning that is spatially aware, enhancing the relevance and specificity of the generated content within designated areas.
## Input types
### Required
- **`clip`**
    - A CLIP model identifier used for encoding the textual prompt into a format suitable for conditioning. It plays a crucial role in aligning the text with the visual content.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`color_mask`**
    - An image serving as a mask, where a specific color defines the region of interest for conditioning. This mask guides where the conditioning should be applied.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
- **`mask_color`**
    - The color value used to identify the region of interest within the color mask. It determines which parts of the image are affected by the conditioning.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`strength`**
    - A scalar value that adjusts the intensity of the conditioning effect within the specified region. It influences how strongly the conditioning is applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`set_cond_area`**
    - Defines the area of conditioning, allowing for either default application or focusing on the bounds of the mask. This choice affects how the conditioning is spatially distributed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`prompt`**
    - The textual prompt that is encoded and integrated with the visual conditioning. It provides the thematic direction for the content generation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The resulting conditioning data, tailored to the specified region and prompt, ready for further processing in content generation.
    - Python dtype: `torch.Tensor`
- **`mask`**
    - Comfy dtype: `MASK`
    - The processed mask that delineates the region of interest for the applied conditioning.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class RegionalConditioningColorMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP", ),
                "color_mask": ("IMAGE",),
                "mask_color": ("STRING", {"multiline": False, "default": "#FFFFFF"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "set_cond_area": (["default", "mask bounds"],),
                "prompt": ("STRING", {"multiline": True, "placeholder": "prompt"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "MASK")
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(clip, color_mask, mask_color, strength, set_cond_area, prompt):
        mask = color_to_mask(color_mask, mask_color)

        conditioning = nodes.CLIPTextEncode().encode(clip, prompt)[0]
        conditioning = nodes.ConditioningSetMask().append(conditioning, mask, set_cond_area, strength)[0]
        return conditioning, mask

```
