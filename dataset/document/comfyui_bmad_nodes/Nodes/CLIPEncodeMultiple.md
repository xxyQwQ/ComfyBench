---
tags:
- CLIP
- CLIPConditioning
- Conditioning
---

# CLIPEncodeMultiple
## Documentation
- Class name: `CLIPEncodeMultiple`
- Category: `Bmad/conditioning`
- Output node: `False`

The CLIPEncodeMultiple node is designed to encode multiple text inputs into a conditioning format using a CLIP model. It iterates over a specified number of inputs, encoding each one separately and aggregating the results into a list of conditionings. This node is useful for scenarios where multiple textual descriptions need to be encoded in parallel and then utilized for further processing or generation tasks.
## Input types
### Required
- **`clip`**
    - The CLIP model used for encoding the text inputs. It is crucial for the text encoding process, affecting the quality and relevance of the generated conditionings.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`inputs_len`**
    - Specifies the number of text inputs to encode. It determines the iteration count for the encoding process, directly influencing the output list's length.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - A list of encoded text inputs, each transformed into a conditioning format suitable for further processing or generation tasks.
    - Python dtype: `List[torch.Tensor]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CLIPEncodeMultiple(nodes.CLIPTextEncode):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",),
            "inputs_len": ("INT", {"default": 9, "min": 0, "max": 32}),
        }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "gen2"
    CATEGORY = conditioning_category_path
    OUTPUT_IS_LIST = (True,)

    def gen2(self, clip, inputs_len, **kwargs):
        conds = []
        for i in range(inputs_len):
            arg_name = get_arg_name_from_multiple_inputs("string", i)
            conds.append(super().encode(clip, kwargs[arg_name])[0])
        return (conds,)

```
