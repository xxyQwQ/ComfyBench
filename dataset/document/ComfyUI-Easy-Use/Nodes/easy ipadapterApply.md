---
tags:
- IPAdapter
---

# Easy Apply IPAdapter
## Documentation
- Class name: `easy ipadapterApply`
- Category: `EasyUse/Adapter`
- Output node: `False`

This node facilitates the application of IPAdapter models to enhance or modify images within a given pipeline, leveraging predefined presets and optional parameters to tailor the adaptation process to specific needs.
## Input types
### Required
- **`model`**
    - The base model to which the IPAdapter will be applied, serving as the foundation for image enhancement or modification.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`image`**
    - The image to be processed or modified by the IPAdapter, acting as the primary subject of adaptation.
    - Comfy dtype: `IMAGE`
    - Python dtype: `Image`
- **`preset`**
    - A predefined configuration set that dictates how the IPAdapter modifies or enhances the image, allowing for standardized or custom adaptation processes.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_strength`**
    - Determines the intensity of the LoRA (Low-Rank Adaptation) applied during the adaptation process, enabling fine-tuning of the model's responsiveness to the IPAdapter's modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`provider`**
    - Specifies the computation provider for the IPAdapter, such as CPU or GPU, influencing the execution environment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`weight`**
    - A factor that influences the overall impact of the IPAdapter on the image, allowing for control over the adaptation intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_faceidv2`**
    - Specifically adjusts the influence of the FaceID v2 feature within the IPAdapter, offering finer control over facial adaptations.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`start_at`**
    - Defines the starting point of the adaptation effect, enabling phased or gradual application of the IPAdapter's modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint for the adaptation effect, allowing for precise control over the extent of modifications applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cache_mode`**
    - Determines the caching strategy for model components, affecting performance and resource utilization.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_tiled`**
    - Indicates whether the adaptation should be applied in a tiled manner, potentially improving performance for large images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
### Optional
- **`attn_mask`**
    - An optional mask that can be applied to focus the IPAdapter's effect on specific areas of the image.
    - Comfy dtype: `MASK`
    - Python dtype: `Mask`
- **`optional_ipadapter`**
    - Allows for the inclusion of an additional, optional IPAdapter to be used in conjunction with the primary one.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `IPAdapter`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced or modified model after applying the IPAdapter, reflecting the adaptations made.
    - Python dtype: `comfy.model_base.Model`
- **`images`**
    - Comfy dtype: `IMAGE`
    - The resulting images after the adaptation process, showcasing the applied modifications or enhancements.
    - Python dtype: `List[Image]`
- **`masks`**
    - Comfy dtype: `MASK`
    - Optional masks generated during the adaptation process, which may be used for further image processing or analysis.
    - Python dtype: `List[Mask]`
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - The IPAdapter instance used in the adaptation process, encapsulating the specific modifications or enhancements applied.
    - Python dtype: `IPAdapter`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ipadapterApply(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        presets = cls().presets
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "cache_mode": (["insightface only", "clip_vision only", "ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
            },

            "optional": {
                "attn_mask": ("MASK",),
                "optional_ipadapter": ("IPADAPTER",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, start_at, end_at, cache_mode, use_tiled, attn_mask=None, optional_ipadapter=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=None, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if use_tiled and preset not in self.faceid_presets:
            if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                self.error()
            cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image, weight, "linear", start_at, end_at, sharpening=0.0, combine_embeds="concat", image_negative=None, attn_mask=attn_mask, clip_vision=None, embeds_scaling='V only')
        else:
            if preset in ['FACEID PLUS V2', 'FACEID PORTRAIT (style transfer)']:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
                model, images = cls().apply_ipadapter(model, ipadapter, start_at=start_at, end_at=end_at, weight=weight, weight_type="linear", combine_embeds="concat", weight_faceidv2=weight_faceidv2, image=image, image_negative=None, clip_vision=None, attn_mask=attn_mask, insightface=None, embeds_scaling='V only')
            else:
                if "IPAdapter" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapter"]
                model, images = cls().apply_ipadapter(model, ipadapter, image, weight, start_at, end_at, weight_type='standard', attn_mask=attn_mask)
        if images is None:
            images = image
        return (model, images, masks, ipadapter,)

```
