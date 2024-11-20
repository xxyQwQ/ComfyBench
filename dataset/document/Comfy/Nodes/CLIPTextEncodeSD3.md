# CLIPTextEncodeSD3
## Documentation
- Class name: `CLIPTextEncodeSD3`
- Category: `advanced/conditioning`
- Output node: `False`

The CLIPTextEncodeSD3 node is designed for advanced text conditioning in generative models, focusing on encoding text inputs using the CLIP model. It supports multiple text inputs and an option for empty padding, facilitating the creation of conditioned inputs for generative tasks.
## Input types
### Required
- **`clip`**
    - The CLIP model instance used for tokenization and encoding of text inputs. It plays a central role in processing the text inputs into a format suitable for conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
- **`clip_l`**
    - A multiline, dynamically promptable string input representing the local text to be encoded. It affects the generation by providing local context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip_g`**
    - A multiline, dynamically promptable string input representing the global text to be encoded. It provides the broader context for the generation task.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`t5xxl`**
    - A multiline, dynamically promptable string input for additional text encoding using a T5 model, enhancing the conditioning with another layer of textual context.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`empty_padding`**
    - Specifies the padding strategy (none or empty_prompt) for handling empty text inputs, influencing the final encoded output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The encoded text output, structured for use in conditioning generative models. It includes both the conditioning tokens and pooled output for comprehensive context.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPTextEncodeSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "clip_g": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "empty_padding": (["none", "empty_prompt"], )
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, clip_l, clip_g, t5xxl, empty_padding):
        no_padding = empty_padding == "none"

        tokens = clip.tokenize(clip_g)
        if len(clip_g) == 0 and no_padding:
            tokens["g"] = []

        if len(clip_l) == 0 and no_padding:
            tokens["l"] = []
        else:
            tokens["l"] = clip.tokenize(clip_l)["l"]

        if len(t5xxl) == 0 and no_padding:
            tokens["t5xxl"] =  []
        else:
            tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], )

```
