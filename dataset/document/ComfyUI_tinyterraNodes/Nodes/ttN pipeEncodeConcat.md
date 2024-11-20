---
tags:
- Image
- Pipeline
---

# pipeEncodeConcat
## Documentation
- Class name: `ttN pipeEncodeConcat`
- Category: `🌏 tinyterra/pipe`
- Output node: `True`

The ttN pipeEncodeConcat node is designed for advanced text processing within a pipeline, focusing on encoding and concatenating text inputs based on specific conditioning. It leverages token normalization and weight interpretation to refine the input text, aiming to enhance the overall quality and relevance of the generated content.
## Input types
### Required
- **`pipe`**
    - Represents the pipeline configuration and state, serving as the foundational context for the node's operations.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`toggle`**
    - A boolean toggle that influences the node's processing behavior, although its specific role is not detailed in the provided context.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
### Optional
- **`positive`**
    - Specifies the positive text input to be encoded and concatenated. It plays a crucial role in guiding the node's output towards desired attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`positive_token_normalization`**
    - Controls the normalization process for tokens in the positive text, affecting the encoding's sensitivity to token variations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`positive_weight_interpretation`**
    - Adjusts the weight interpretation for the positive text, impacting the emphasis placed on different aspects of the input during encoding.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative`**
    - Defines the negative text input for encoding and concatenation. This input helps in steering the output away from undesired attributes or themes.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`negative_token_normalization`**
    - Manages the normalization of tokens in the negative text, influencing how the encoding perceives token differences.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`negative_weight_interpretation`**
    - Modifies the weight interpretation for the negative text, altering the focus on various elements of the input during encoding.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`optional_positive_from`**
    - Optional input specifying an alternative source of positive conditioning, allowing for dynamic adjustment of the encoding process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`optional_negative_from`**
    - Optional input indicating an alternative source of negative conditioning, enabling flexible modification of the encoding approach.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `dict`
- **`optional_clip`**
    - An optional CLIP model parameter that can be used to further refine the encoding process, offering an additional layer of customization.
    - Comfy dtype: `CLIP`
    - Python dtype: `dict`
## Output types
- **`pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The updated pipeline configuration after processing, reflecting changes made by the node.
    - Python dtype: `dict`
- **`positive`**
    - Comfy dtype: `CONDITIONING`
    - The enhanced positive conditioning output, resulting from the encoding and concatenation process.
    - Python dtype: `dict`
- **`negative`**
    - Comfy dtype: `CONDITIONING`
    - The refined negative conditioning output, produced through the encoding and concatenation efforts.
    - Python dtype: `dict`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The possibly modified CLIP model parameter, adjusted during the node's operation.
    - Python dtype: `dict`
- **`ui`**
    - Provides a user interface element that displays the new text generated after encoding and concatenation.
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeEncodeConcat:
    version = '1.0.2'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "pipe": ("PIPE_LINE",),
                    "toggle": ([True, False],),
                    },
                "optional": {
                    "positive": ("STRING", {"default": "Positive","multiline": True}),
                    "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                    "negative": ("STRING", {"default": "Negative","multiline": True}),
                    "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                    "optional_positive_from": ("CONDITIONING",),
                    "optional_negative_from": ("CONDITIONING",),
                    "optional_clip": ("CLIP",),
                    },
                "hidden": {
                    "ttNnodeVersion": ttN_pipeEncodeConcat.version, "my_unique_id": "UNIQUE_ID"
                    },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING", "CLIP")
    RETURN_NAMES = ("pipe", "positive", "negative", "clip")
    FUNCTION = "concat"

    CATEGORY = "🌏 tinyterra/pipe"

    def concat(self, toggle, positive_token_normalization, positive_weight_interpretation,
               negative_token_normalization, negative_weight_interpretation,
                 pipe=None, positive='', negative='', seed=None, my_unique_id=None, optional_positive_from=None, optional_negative_from=None, optional_clip=None):
        
        if toggle == False:
            return (pipe, pipe["positive"], pipe["negative"], pipe["clip"])
        
        positive_from = optional_positive_from if optional_positive_from is not None else pipe["positive"] 
        negative_from = optional_negative_from if optional_negative_from is not None else pipe["negative"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        new_text = ''

        def enConcatConditioning(text, token_normalization, weight_interpretation, conditioning_from, new_text):
            out = []
            if "__" in text:
                text = loader.nsp_parse(text, pipe["seed"], title="encodeConcat", my_unique_id=my_unique_id)
                new_text += text

            conditioning_to, pooled = advanced_encode(samp_clip, text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            conditioning_to = [[conditioning_to, {"pooled_output": pooled}]]

            if len(conditioning_from) > 1:
                ttNl.warn("encode and concat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to")

            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from),1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)

            return out

        pos, neg = None, None
        if positive not in ['', None, ' ']:
            pos = enConcatConditioning(positive, positive_token_normalization, positive_weight_interpretation, positive_from, new_text)
        if negative not in ['', None, ' ']:
            neg = enConcatConditioning(negative, negative_token_normalization, negative_weight_interpretation, negative_from, new_text)

        pos = pos if pos is not None else pipe["positive"]
        neg = neg if neg is not None else pipe["negative"]
        
        new_pipe = {
                "model": pipe["model"],
                "positive": pos,
                "negative": neg,
                "vae": pipe["vae"],
                "clip": samp_clip,

                "samples": pipe["samples"],
                "images": pipe["images"],
                "seed": pipe["seed"],

                "loader_settings": pipe["loader_settings"],
            }
        del pipe

        return (new_pipe, new_pipe["positive"], new_pipe["negative"], samp_clip, { "ui": { "string": new_text } } )

```
