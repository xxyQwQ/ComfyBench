---
tags:
- Conditioning
---

# Concat Conditionings
## Documentation
- Class name: `ImpactConcatConditionings`
- Category: `ImpactPack/Util`
- Output node: `False`

The ImpactConcatConditionings node is designed to concatenate multiple conditioning inputs into a single conditioning output. This process is essential for combining various conditioning elements to enhance or specify the generation process further.
## Input types
### Required
- **`conditioning1`**
    - The primary conditioning input. This input serves as the base to which additional conditionings are concatenated, influencing the overall output by combining multiple conditioning aspects.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Any]]`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The concatenated conditioning output, which combines the input conditionings into a unified form, enhancing the generation process.
    - Python dtype: `List[Tuple[torch.Tensor, Any]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class ConcatConditionings:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "conditioning1": ("CONDITIONING", ),
                     },
                }

    TOOLTIPS = {
        "input": {
            "conditioning1": "input conditionings. (Connecting to the input slot increases the number of additional slots.)",
        },
        "output": ("Concatenated conditioning", )
    }

    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Util"

    @staticmethod
    def doit(**kwargs):
        conditioning_to = list(kwargs.values())[0]

        for k, conditioning_from in list(kwargs.items())[1:]:
            out = []
            if len(conditioning_from) > 1:
                print("Warning: ConcatConditionings {k} contains more than 1 cond, only the first one will actually be applied to conditioning1.")

            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from), 1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)

            conditioning_to = out

        return (out, )

```
