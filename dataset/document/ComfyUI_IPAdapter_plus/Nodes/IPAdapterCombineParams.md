---
tags:
- IPAdapter
---

# IPAdapter Combine Params
## Documentation
- Class name: `IPAdapterCombineParams`
- Category: `ipadapter/params`
- Output node: `False`

This node is designed to combine multiple sets of parameters for IP adapters, allowing for the flexible integration of various input parameters into a single, unified set. It supports the aggregation of images, attention masks, weights, weight types, and start and end points from up to five different parameter sets, facilitating complex parameter management and customization in image processing tasks.
## Input types
### Required
- **`params_i`**
    - A set of parameters to be combined. This generalized parameter name represents any of the up to five sets of parameters (params_1 to params_5) that can be provided. Each set contributes to the overall aggregation, enriching the combined parameters with its values. The index 'i' ranges from 1 to 5, indicating the position of the parameter set in the combination process.
    - Comfy dtype: `IPADAPTER_PARAMS`
    - Python dtype: `Dict[str, Any]`
### Optional
## Output types
- **`ipadapter_params`**
    - Comfy dtype: `IPADAPTER_PARAMS`
    - The resulting combined set of parameters, encompassing aggregated values from up to five input parameter sets. This unified parameter set is ready for use in further image processing tasks.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class IPAdapterCombineParams:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "params_1": ("IPADAPTER_PARAMS",),
            "params_2": ("IPADAPTER_PARAMS",),
        }, "optional": {
            "params_3": ("IPADAPTER_PARAMS",),
            "params_4": ("IPADAPTER_PARAMS",),
            "params_5": ("IPADAPTER_PARAMS",),
        }}

    RETURN_TYPES = ("IPADAPTER_PARAMS",)
    FUNCTION = "combine"
    CATEGORY = "ipadapter/params"

    def combine(self, params_1, params_2, params_3=None, params_4=None, params_5=None):
        ipadapter_params = {
            "image": params_1["image"] + params_2["image"],
            "attn_mask": params_1["attn_mask"] + params_2["attn_mask"],
            "weight": params_1["weight"] + params_2["weight"],
            "weight_type": params_1["weight_type"] + params_2["weight_type"],
            "start_at": params_1["start_at"] + params_2["start_at"],
            "end_at": params_1["end_at"] + params_2["end_at"],
        }

        if params_3 is not None:
            ipadapter_params["image"] += params_3["image"]
            ipadapter_params["attn_mask"] += params_3["attn_mask"]
            ipadapter_params["weight"] += params_3["weight"]
            ipadapter_params["weight_type"] += params_3["weight_type"]
            ipadapter_params["start_at"] += params_3["start_at"]
            ipadapter_params["end_at"] += params_3["end_at"]
        if params_4 is not None:
            ipadapter_params["image"] += params_4["image"]
            ipadapter_params["attn_mask"] += params_4["attn_mask"]
            ipadapter_params["weight"] += params_4["weight"]
            ipadapter_params["weight_type"] += params_4["weight_type"]
            ipadapter_params["start_at"] += params_4["start_at"]
            ipadapter_params["end_at"] += params_4["end_at"]
        if params_5 is not None:
            ipadapter_params["image"] += params_5["image"]
            ipadapter_params["attn_mask"] += params_5["attn_mask"]
            ipadapter_params["weight"] += params_5["weight"]
            ipadapter_params["weight_type"] += params_5["weight_type"]
            ipadapter_params["start_at"] += params_5["start_at"]
            ipadapter_params["end_at"] += params_5["end_at"]

        return (ipadapter_params, )

```
