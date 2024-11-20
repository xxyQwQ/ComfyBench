---
tags:
- CLIP
- CLIPConditioning
- Conditioning
---

# CLIPEncodeMultipleAdvanced
## Documentation
- Class name: `CLIPEncodeMultipleAdvanced`
- Category: `Bmad/conditioning`
- Output node: `False`

This node is designed for advanced text encoding using the CLIP model, capable of processing multiple inputs simultaneously. It extends the functionality of a standard CLIP text encoding by allowing for a customizable number of inputs, each potentially undergoing a unique encoding process based on token normalization and weight interpretation. The node is optimized for scenarios requiring batch processing of text inputs for conditioning generation models, making it a versatile tool in the generation of conditioned outputs.
## Input types
### Required
- **`clip`**
    - The CLIP model used for encoding the text inputs. It plays a crucial role in determining the quality and relevance of the encoded vectors.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`token_normalization`**
    - A flag indicating whether or not to normalize the tokens during the encoding process. This affects the consistency and comparability of the encoded vectors.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`weight_interpretation`**
    - A parameter that influences how the weights are interpreted during the encoding process, affecting the emphasis on different aspects of the input text.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`inputs_len`**
    - Specifies the number of text inputs to be encoded. This allows for dynamic adjustment of batch sizes for encoding, accommodating a variable number of inputs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - A list of conditioning vectors generated from the input texts, tailored for use in generation models requiring conditioned inputs.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPEncodeMultipleAdvanced(AdvancedCLIPTextEncode):
    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()  # TODO should refactor Grid class above to this too, so if original is changed, all the new options are added there too
        types["required"].pop("text")
        types["required"]["inputs_len"] = ("INT", {"default": 9, "min": 0, "max": 32})
        return types

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "gen2"
    CATEGORY = conditioning_category_path
    OUTPUT_IS_LIST = (True,)

    def gen2(self, clip, token_normalization, weight_interpretation, inputs_len, **kwargs):
        conds = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("string", i)
            conds.append(
                super().encode(clip, kwargs[arg_name], token_normalization, weight_interpretation, 'disable')[0])
        return (conds,)

```
