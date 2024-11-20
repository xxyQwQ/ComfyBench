---
tags:
- CLIP
- CLIPConditioning
- CLIPTextEncoding
- Conditioning
---

# 🔧 SDXL CLIPTextEncode
## Documentation
- Class name: `CLIPTextEncodeSDXL+`
- Category: `essentials/conditioning`
- Output node: `False`

This node is designed to encode text inputs using a CLIP model, specifically tailored for the Stable Diffusion XL framework. It processes text inputs to generate conditioning vectors and pooled outputs that are optimized for image generation tasks, incorporating additional parameters such as aesthetic scores and dimensions to fine-tune the generated images.
## Input types
### Required
- **`width`**
    - Defines the width of the target image in pixels, affecting the aspect ratio and resolution of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the target image in pixels, impacting the aspect ratio and resolution of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`size_cond_factor`**
    - Specifies the factor by which the dimensions of the target image are scaled, affecting the detail and scale of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`text`**
    - The text input to be encoded, serving as the basis for generating the conditioning vector and influencing the content of the generated image.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - The CLIP model used for text tokenization and encoding, central to generating the conditioning vectors.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Outputs a conditioning vector and associated metadata, including the aesthetic score and image dimensions, tailored for image generation.
    - Python dtype: `list`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPTextEncodeSDXLSimplified:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "size_cond_factor": ("INT", {"default": 4, "min": 1, "max": 16 }),
            "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": ""}),
            "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute"
    CATEGORY = "essentials/conditioning"

    def execute(self, clip, width, height, size_cond_factor, text):
        crop_w = 0
        crop_h = 0
        width = width*size_cond_factor
        height = height*size_cond_factor
        target_width = width
        target_height = height
        text_g = text_l = text

        tokens = clip.tokenize(text_g)
        tokens["l"] = clip.tokenize(text_l)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_w, "crop_h": crop_h, "target_width": target_width, "target_height": target_height}]], )

```
