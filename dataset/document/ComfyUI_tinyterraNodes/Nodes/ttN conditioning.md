---
tags:
- Conditioning
---

# tinyConditioning
## Documentation
- Class name: `ttN conditioning`
- Category: `🌏 tinyterra/base`
- Output node: `False`

The node focuses on applying conditioning transformations to input data, leveraging advanced encoding techniques to enhance or modify the data based on specified conditions. It abstracts complex conditioning logic into a more accessible interface for users, aiming to streamline the process of data manipulation and enrichment.
## Input types
### Required
- **`model`**
    - Specifies the model to be used for conditioning, playing a central role in how the input data is processed and transformed.
    - Comfy dtype: `MODEL`
    - Python dtype: `ModelPatcher | None`
- **`clip`**
    - Defines the CLIP model parameters to be used in conjunction with the main model for conditioning, affecting the outcome of the transformation.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP | None`
- **`loras`**
    - Specifies the LoRA configurations to be applied, influencing the conditioning process by adjusting the model's behavior.
    - Comfy dtype: `STRING`
    - Python dtype: `str | None`
- **`positive`**
    - The positive text input for conditioning, used to guide the model towards desired attributes or outcomes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`positive_token_normalization`**
    - Normalization method for the positive text, affecting how the text is processed before conditioning.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive_weight_interpretation`**
    - Determines how the positive text's weight is interpreted during the conditioning process, influencing the model's adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative`**
    - The negative text input for conditioning, used to steer the model away from undesired attributes or outcomes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_token_normalization`**
    - Normalization method for the negative text, affecting how the text is processed before conditioning.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative_weight_interpretation`**
    - Determines how the negative text's weight is interpreted during the conditioning process, influencing the model's adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`optional_lora_stack`**
    - An optional stack of LoRA configurations to be applied, providing additional customization to the conditioning process.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `List[Tuple[str, int, float]] | None`
- **`prepend_positive`**
    - Text to prepend to the positive input, further customizing the conditioning process.
    - Comfy dtype: `STRING`
    - Python dtype: `str | None`
- **`prepend_negative`**
    - Text to prepend to the negative input, further customizing the conditioning process.
    - Comfy dtype: `STRING`
    - Python dtype: `str | None`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - Returns the conditioned model after applying the specified transformations.
    - Python dtype: `ModelPatcher | None`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the positive conditioning effects applied to the model.
    - Python dtype: `torch.Tensor`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - Outputs the negative conditioning effects applied to the model.
    - Python dtype: `torch.Tensor`
- **`clip`**
    - Comfy dtype: `CLIP`
    - Returns the CLIP model parameters used during the conditioning process.
    - Python dtype: `CLIP | None`
- **`pos_string`**
    - Comfy dtype: `STRING`
    - Outputs the final positive string after conditioning, reflecting the combined and processed input.
    - Python dtype: `str`
- **`neg_string`**
    - Comfy dtype: `STRING`
    - Outputs the final negative string after conditioning, reflecting the combined and processed input.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ttN_conditioning:
    version = '1.0.0'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "model": ("MODEL",),
                        "clip": ("CLIP",),

                        "loras": ("STRING", {"placeholder": "<lora:loraName:weight:optClipWeight>", "multiline": True}),

                        "positive": ("STRING", {"default": "Positive","multiline": True, "dynamicPrompts": True}),
                        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True, "dynamicPrompts": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                        },                
                "optional": {
                    "optional_lora_stack": ("LORA_STACK",),
                    "prepend_positive": ("STRING", {"default": None, "forceInput": True}),
                    "prepend_negative": ("STRING", {"default": None, "forceInput": True}),
                    },
                "hidden": {"ttNnodeVersion": ttN_conditioning.version, "my_unique_id": "UNIQUE_ID"},}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "clip", "pos_string", "neg_string")

    FUNCTION = "condition"
    CATEGORY = "🌏 tinyterra/base"

    def condition(self, model, clip, loras,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       optional_lora_stack=None, prepend_positive=None, prepend_negative=None,
                       my_unique_id=None):

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                model, clip = loader.load_lora(lora[0], model, clip, lora[1], lora[2])
                
        if loras not in [None, "None"]:
            model, clip = loader.load_lora_text(loras, model, clip)

        positive_embedding = loader.embedding_encode(positive, positive_token_normalization, positive_weight_interpretation, clip, title='ttN Conditioning Positive', my_unique_id=my_unique_id, prepend_text=prepend_positive)
        negative_embedding = loader.embedding_encode(negative, negative_token_normalization, negative_weight_interpretation, clip, title='ttN Conditioning Negative', my_unique_id=my_unique_id, prepend_text=prepend_negative)

        final_positive = (prepend_positive + ' ' if prepend_positive else '') + (positive + ' ' if positive else '')
        final_negative = (prepend_negative + ' ' if prepend_negative else '') + (negative + ' ' if negative else '')

        return (model, positive_embedding, negative_embedding, clip, final_positive, final_negative)

```
