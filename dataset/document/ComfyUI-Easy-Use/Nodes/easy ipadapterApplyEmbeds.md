---
tags:
- IPAdapter
---

# Easy Apply IPAdapter (Embeds)
## Documentation
- Class name: `easy ipadapterApplyEmbeds`
- Category: `EasyUse/Adapter`
- Output node: `False`

The node 'easy ipadapterApplyEmbeds' is designed to apply embedding transformations to models using IPAdapter, facilitating the integration of positional and negative embeddings into the model's processing pipeline. It abstracts the complexity of embedding manipulation, offering a streamlined approach to enhance model performance with custom embeddings.
## Input types
### Required
- **`model`**
    - Specifies the model to which embeddings will be applied, serving as the foundational structure for embedding integration.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`clip_vision`**
    - Indicates whether to apply CLIP vision embeddings, influencing the model's visual understanding and processing.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `bool`
- **`ipadapter`**
    - The IPAdapter instance used for embedding transformations, central to the embedding application process.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`pos_embed`**
    - The positive embeddings to be applied, enhancing the model's positive feature recognition.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `list`
- **`weight`**
    - Defines the weight of the embeddings, influencing their impact on the model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - Specifies the type of weight applied to the embeddings, affecting their integration into the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`start_at`**
    - Determines the starting point for embedding application, allowing for precise control over when embeddings influence the model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`end_at`**
    - Sets the endpoint for embedding application, defining the scope of embedding influence on the model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`embeds_scaling`**
    - Defines the scaling method for embeddings, affecting how embeddings are integrated and weighted within the model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`neg_embed`**
    - The negative embeddings to be applied, enhancing the model's negative feature recognition.
    - Comfy dtype: `EMBEDS`
    - Python dtype: `list`
- **`attn_mask`**
    - An optional attention mask for more targeted embedding application, providing additional control over how embeddings affect the model.
    - Comfy dtype: `MASK`
    - Python dtype: `list`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The enhanced model with applied embeddings, reflecting the integration of positive and negative embeddings.
    - Python dtype: `str`
- **`ipadapter`**
    - Comfy dtype: `IPADAPTER`
    - The IPAdapter instance after embedding application, indicating the successful integration of embeddings.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ipadapterApplyEmbeds(ipadapter):
    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def INPUT_TYPES(cls):
        ipa_cls = cls()
        weight_types = ipa_cls.weight_types
        return {
            "required": {
                "model": ("MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "ipadapter": ("IPADAPTER",),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", {"default": 1.0, "min": -1, "max": 3, "step": 0.05}),
                "weight_type": (weight_types,),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],),
            },

            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL", "IPADAPTER",)
    RETURN_NAMES = ("model", "ipadapter", )
    CATEGORY = "EasyUse/Adapter"
    FUNCTION = "apply"

    def apply(self, model, ipadapter, clip_vision, pos_embed, weight, weight_type, start_at, end_at, embeds_scaling, attn_mask=None, neg_embed=None,):
        if "IPAdapterEmbeds" not in ALL_NODE_CLASS_MAPPINGS:
            self.error()

        cls = ALL_NODE_CLASS_MAPPINGS["IPAdapterEmbeds"]
        model, image = cls().apply_ipadapter(model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=neg_embed, attn_mask=attn_mask, clip_vision=clip_vision, embeds_scaling=embeds_scaling)

        return (model, ipadapter)

```
