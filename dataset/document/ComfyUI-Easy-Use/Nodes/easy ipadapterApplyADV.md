---
tags:
- IPAdapter
---

# Easy Apply IPAdapter (Advanced)
## Documentation
- Class name: `easy ipadapterApplyADV`
- Category: `EasyUse/Adapter`
- Output node: `False`

This node specializes in applying advanced IPAdapter configurations to models, enhancing their capabilities with custom style, composition, and embedding adjustments. It leverages a sophisticated set of parameters to fine-tune the integration of IPAdapter into the model's processing pipeline, offering a high degree of customization for generating or modifying images.
## Input types
### Required
- **`model`**
    - The model to which the IPAdapter will be applied, serving as the foundation for the advanced customization and enhancements.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`image`**
    - The image to be processed, serving as the input for the IPAdapter's style and composition adjustments.
    - Comfy dtype: `IMAGE`
    - Python dtype: `IMAGE`
- **`preset`**
    - Defines the preset configuration for the IPAdapter, guiding its application strategy.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_strength`**
    - Adjusts the strength of the LoRA adjustments within the IPAdapter, affecting the depth of customization.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`provider`**
    - Specifies the provider for the model or service, influencing the selection of processing resources.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`weight`**
    - The overall weight influencing the application of the IPAdapter, affecting the output's visual characteristics.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_faceidv2`**
    - Specific weight for FaceID v2 adjustments, tailoring the facial recognition enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Determines the type of weighting applied, affecting the balance between different adjustment types.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`combine_embeds`**
    - Determines whether embeddings should be combined, impacting the integration of multiple embeddings into the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`start_at`**
    - Specifies the starting point within the model's layers for applying the IPAdapter, allowing for targeted enhancements.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Defines the end point within the model's layers for the application of the IPAdapter, enabling precise control over the extent of modifications.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Controls the scaling of embeddings, adjusting the impact of embeddings on the model's output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`cache_mode`**
    - Controls the caching strategy for the IPAdapter, optimizing performance and resource usage.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`use_tiled`**
    - Indicates whether to apply the IPAdapter in a tiled manner, affecting the processing of large images.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`use_batch`**
    - Specifies whether to process images in batches, improving efficiency for multiple inputs.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`sharpening`**
    - Adjusts the sharpness of the output images, enhancing detail and clarity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
### Optional
- **`image_negative`**
    - unknown
    - Comfy dtype: `IMAGE`
    - Python dtype: `unknown`
- **`attn_mask`**
    - An optional attention mask to focus the IPAdapter's adjustments on specific areas, enhancing the relevance of modifications.
    - Comfy dtype: `MASK`
    - Python dtype: `AttnMask`
- **`clip_vision`**
    - unknown
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `unknown`
- **`optional_ipadapter`**
    - An optional IPAdapter parameter for additional customization and flexibility in the application process.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `IPAdapter`
- **`layer_weights`**
    - Specifies the weights for different layers within the model, allowing for fine-tuned adjustments.
    - Comfy dtype: `STRING`
    - Python dtype: `Dict[str, float]`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model with the advanced IPAdapter configurations applied, reflecting the customized adjustments.
    - Python dtype: `comfy.model_base.Model`
- **`images`**
    - Comfy dtype: `IMAGE`
    - The images generated or modified by the IPAdapter, showcasing the applied adjustments.
    - Python dtype: `List[IMAGE]`
- **`masks`**
    - Comfy dtype: `MASK`
    - The masks applied during the IPAdapter process, indicating areas of focus or exclusion.
    - Python dtype: `List[Optional[MASK]]`
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - The IPAdapter instance after application, encapsulating the advanced configurations and customizations.
    - Python dtype: `IPAdapter`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ipadapterApplyAdvanced(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        presets = ipa_cls.presets
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "preset": (presets,),
                "lora_strength": ("FLOAT", {"default": 0.6, "min": 0, "max": 1, "step": 0.01}),
                "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"],),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_faceidv2": ("FLOAT", {"default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (weight_types,),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
                "cache_mode": (["insightface only", "clip_vision only","ipadapter only", "all", "none"], {"default": "all"},),
                "use_tiled": ("BOOLEAN", {"default": False},),
                "use_batch": ("BOOLEAN", {"default": False},),
                "sharpening": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },

            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "optional_ipadapter": ("IPADAPTER",),
                "layer_weights": ("STRING", {"default": "", "multiline": True, "placeholder": "Mad Scientist Layer Weights"}),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", "IPADAPTER",)
    RETURN_NAMES = ("model", "images", "masks", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, image, preset, lora_strength, provider, weight, weight_faceidv2, weight_type, combine_embeds, start_at, end_at, embeds_scaling, cache_mode, use_tiled, use_batch, sharpening, weight_style=1.0, weight_composition=1.0, image_style=None, image_composition=None, expand_style=False, image_negative=None, clip_vision=None, attn_mask=None, optional_ipadapter=None, layer_weights=None):
        images, masks = image, [None]
        model, ipadapter = self.load_model(model, preset, lora_strength, provider, clip_vision=clip_vision, optional_ipadapter=optional_ipadapter, cache_mode=cache_mode)
        if layer_weights:
            if "IPAdapterMS" not in ALL_NODE_CLASS_MAPPINGS:
                self.error()
            cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
            model, images = cls().apply_ipadapter(model, ipadapter, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, combine_embeds=combine_embeds, weight_faceidv2=weight_faceidv2, image=image, image_negative=image_negative, weight_style=weight_style, weight_composition=weight_composition, image_style=image_style, image_composition=image_composition, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling, layer_weights=layer_weights)
        elif use_tiled:
            if use_batch:
                if "IPAdapterTiledBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiledBatch"]
            else:
                if "IPAdapterTiled" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterTiled"]
            model, images, masks = cls().apply_tiled(model, ipadapter, image=image, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, sharpening=sharpening, combine_embeds=combine_embeds, image_negative=image_negative, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)
        else:
            if use_batch:
                if "IPAdapterBatch" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterBatch"]
            else:
                if "IPAdapterAdvanced" not in ALL_NODE_CLASS_MAPPINGS:
                    self.error()
                cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]
            model, images = cls().apply_ipadapter(model, ipadapter, weight=weight, weight_type=weight_type, start_at=start_at, end_at=end_at, combine_embeds=combine_embeds, weight_faceidv2=weight_faceidv2, image=image, image_negative=image_negative, weight_style=1.0, weight_composition=1.0, image_style=image_style, image_composition=image_composition, expand_style=expand_style, clip_vision=clip_vision, attn_mask=attn_mask, insightface=None, embeds_scaling=embeds_scaling)
        if images is None:
            images = image
        return (model, images, masks, ipadapter)

```
