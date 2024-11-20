---
tags:
- IPAdapter
---

# IPAdapter Embeds
## Documentation
- Class name: `IPAdapterEmbeds`
- Category: `ipadapter/embeds`
- Output node: `False`

This node is designed for embedding generation within the IPAdapter framework. It focuses on creating embeddings from given inputs, which can then be utilized in various image processing and enhancement tasks, emphasizing the adaptability and enhancement of images through embedding manipulation.
## Input types
### Required
- **`model`**
    - unknown
    - Comfy dtype: `MODEL`
    - Python dtype: `unknown`
- **`ipadapter`**
    - Specifies the IPAdapter to be used for embedding generation, indicating the specific adapter configuration.
    - Comfy dtype: `IPADAPTER`
    - Python dtype: `str`
- **`pos_embed`**
    - unknown
    - Comfy dtype: `EMBEDS`
    - Python dtype: `unknown`
- **`weight`**
    - A floating-point value that adjusts the influence of the embedding in the overall processing, allowing for fine-tuning of the embedding's impact.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`weight_type`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
- **`start_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`end_at`**
    - unknown
    - Comfy dtype: `FLOAT`
    - Python dtype: `unknown`
- **`embeds_scaling`**
    - unknown
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `unknown`
### Optional
- **`neg_embed`**
    - unknown
    - Comfy dtype: `EMBEDS`
    - Python dtype: `unknown`
- **`attn_mask`**
    - unknown
    - Comfy dtype: `MASK`
    - Python dtype: `unknown`
- **`clip_vision`**
    - An optional parameter that allows for the inclusion of CLIP vision features, enhancing the embedding's adaptability to various contexts.
    - Comfy dtype: `CLIP_VISION`
    - Python dtype: `torch.Tensor`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The output model enhanced with generated embeddings, showcasing the adaptability and enhancement capabilities of the IPAdapter framework.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterEmbeds:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter/embeds"

    def apply_ipadapter(self, model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=None, attn_mask=None, clip_vision=None, embeds_scaling='V only'):
        ipa_args = {
            "pos_embed": pos_embed,
            "neg_embed": neg_embed,
            "weight": weight,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "embeds_scaling": embeds_scaling,
            "unfold_batch": self.unfold_batch,
        }

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None and neg_embed is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        return ipadapter_execute(model.clone(), ipadapter_model, clip_vision, **ipa_args)

```
