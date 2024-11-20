---
tags:
- ImageScaling
- ImageUpscaling
- Upscale
---

# Clarity AI Upscaler
## Documentation
- Class name: `Clarity AI Upscaler`
- Category: `Clarity AI`
- Output node: `False`

The Clarity AI Upscaler node is designed to enhance image quality through upscaling, utilizing Clarity AI's API to process images based on various artistic and technical parameters. It allows for customization of the upscaling process through attributes like creativity, resemblance, dynamic, fractality, and style, aiming to produce higher-resolution images that meet specific aesthetic or technical requirements.
## Input types
### Required
- **`image`**
    - The primary image input for upscaling. This is a required parameter that defines the image to be processed and enhanced.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`prompt`**
    - An optional text prompt that can guide the upscaling process, allowing for more creative or specific enhancements based on textual descriptions.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`creativity`**
    - Adjusts the level of creative interpretation applied during the upscaling process. Affects how the AI interprets and enhances the image details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`resemblance`**
    - Controls the degree to which the upscaled image should resemble the original, balancing between fidelity and enhancement.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`dynamic`**
    - Modifies the dynamic range of the upscaled image, potentially enhancing contrast and depth.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`fractality`**
    - Influences the complexity and detail of textures in the upscaled image, adding more intricate patterns or smoothing out details.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`style`**
    - Specifies the artistic style to be applied during upscaling, such as default, portrait, or anime, allowing for stylized enhancements.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`scale_factor`**
    - Determines the factor by which the image will be upscaled, offering various predefined scaling options.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`api_key_override`**
    - Allows for the use of a specific API key for the upscaling process, overriding the default key if necessary.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The enhanced, upscaled version of the input image, processed through Clarity AI's upscaling API.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ClarityAIUpscaler(ClarityBase):
    API_ENDPOINT = ""
    POLL_ENDPOINT  = ""
    ACCEPT = "image/*"
    INPUT_SPEC = {
        "required": {
            "image": ("IMAGE",),
        },
        "optional": {
            "prompt": ("STRING", {"multiline": True}),
            "creativity": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 1}),
            "resemblance": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 1}),
            "dynamic": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 1}),
            "fractality": ("FLOAT", {"default": 0, "min": -10, "max": 10, "step": 1}),
            "style": (["default", "portrait", "anime"],),
            "scale_factor": (["2", "4", "6", "8", "10", "12", "14", "16"],),
            "api_key_override": ("STRING", {"multiline": False}),
        }
    }

```
