---
tags:
- IPAdapter
---

# Easy Apply IPAdapter (StyleComposition)
## Documentation
- Class name: `easy ipadapterStyleComposition`
- Category: `EasyUse/Adapter`
- Output node: `False`

This node facilitates the application of style and composition adjustments to images using an IPAdapter. It abstracts the complexity of applying various style and composition weights and parameters, enabling users to easily enhance or modify the visual attributes of images.
## Input types
### Required
- **`model`**
    - Specifies the model to which the IPAdapter adjustments will be applied, serving as the foundation for style and composition enhancements.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`image_style`**
    - The image that provides the style reference, guiding the stylistic adjustments to be applied to the target image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`preset`**
    - Defines the preset configuration for the IPAdapter, setting a baseline for style and composition adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`weight_style`**
    - Specifies the weight for style adjustments, influencing the intensity of style application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_composition`**
    - Specifies the weight for composition adjustments, influencing the intensity of composition application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`expand_style`**
    - Indicates whether to expand the style adjustments beyond the initial parameters, allowing for more extensive stylistic modifications.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`combine_embeds`**
    - Determines how style and composition embeddings are combined, affecting the final visual outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Specifies the starting point for applying style and composition adjustments, allowing for phased application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Specifies the ending point for applying style and composition adjustments, defining the scope of application.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Controls the scaling of embeddings, affecting the intensity and focus of style and composition adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`cache_mode`**
    - Determines the caching strategy for the IPAdapter, optimizing performance and resource usage.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`image_composition`**
    - The image that serves as a composition reference, guiding the compositional adjustments to be applied to the target image.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`image_negative`**
    - An optional image that provides a negative style reference, used to guide adjustments by indicating undesired stylistic attributes.
    - Comfy dtype: `IMAGE`
    - Python dtype: `str`
- **`attn_mask`**
    - An optional attention mask, used to focus or exclude specific areas from style and composition adjustments.
    - Comfy dtype: `MASK`
    - Python dtype: `str`
- **`clip_vision`**
    - Indicates whether to use CLIP's vision capabilities for guiding the style and composition adjustments, enhancing the alignment between the image's content and the desired stylistic attributes.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `bool`
- **`optional_ipadapter`**
    - An optional IPAdapter instance, allowing for the use of a pre-configured adapter for style and composition adjustments.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model with applied style and composition adjustments, ready for further processing or generation tasks.
    - Python dtype: `str`
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - The IPAdapter instance configured with the specified style and composition adjustments, encapsulating the transformation logic.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ipadapterStyleComposition(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        normal_presets = ipa_cls.normal_presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image_style": ("IMAGE",),
                "preset": (normal_presets,),
                "weight_style": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "weight_composition": ("FLOAT", {"default": 1.0, "min": -1, "max": 5, "step": 0.05}),
                "expand_style": ("BOOLEAN", {"default": False}),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"],
                               {"default": "all"},),
            },
            "optional": {
                "image_composition": ("IMAGE",),
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    CATEGORY = "EasyUse/Adapter"

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter",)
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, preset, weight_style, weight_composition, expand_style, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, image_style=None , image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None):
        model, ipadapter = self.load_model(model, preset, 0, 'CPU', clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)

        if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()
        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]

        model, image = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight_style=weight_style, weight_composition=weight_composition, weight_type='linear', combine_embeds=combine_embeds, weight_faceidv2=weight_composition, image_style=image_style, image_composition=image_composition, image_negative=image_negative, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling)
        return (model, ipadapter)

```
