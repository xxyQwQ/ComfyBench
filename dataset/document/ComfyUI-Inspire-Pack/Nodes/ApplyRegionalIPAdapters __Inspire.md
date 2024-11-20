---
tags:
- IPAdapter
---

# Apply Regional IPAdapters (Inspire)
## Documentation
- Class name: `ApplyRegionalIPAdapters __Inspire`
- Category: `InspirePack/Regional`
- Output node: `False`

This node is designed to apply a series of regional IP adapter transformations to a model, leveraging a pipeline of IP adapters and regional IP adapters to enhance or modify the model's capabilities based on regional information.
## Input types
### Required
- **`ipadapter_pipe`**
    - A tuple containing the main IP adapter, model, clip vision module, insightface module, and lora loader, which collectively form the pipeline through which the regional IP adapters are applied.
    - Comfy dtype: `IPADAPTER_PIPE`
    - Python dtype: `Tuple[IPADAPTER_PIPE]`
- **`regional_ipadapter1`**
    - A regional IP adapter that is applied to the model as part of the transformation process.
    - Comfy dtype: `REGIONAL_IPADAPTER`
    - Python dtype: `REGIONAL_IPADAPTER`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The transformed model after applying the regional IP adapters.
    - Python dtype: `MODEL`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ApplyRegionalIPAdapters:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ipadapter_pipe": ("IPADAPTER_PIPE",),
                    "regional_ipadapter1": ("REGIONAL_IPADAPTER", ),
                    },
                }

    RETURN_TYPES = ("MODEL", )
    FUNCTION = "doit"

    CATEGORY = "InspirePack/Regional"

    @staticmethod
    def doit(**kwargs):
        ipadapter_pipe = kwargs['ipadapter_pipe']
        ipadapter, model, clip_vision, insightface, lora_loader = ipadapter_pipe

        del kwargs['ipadapter_pipe']

        for k, v in kwargs.items():
            ipadapter_pipe = ipadapter, model, clip_vision, insightface, lora_loader
            model = v.doit(ipadapter_pipe)

        return (model, )

```
