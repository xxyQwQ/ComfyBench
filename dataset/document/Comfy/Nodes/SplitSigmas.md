---
tags:
- SigmaScheduling
---

# SplitSigmas
## Documentation
- Class name: `SplitSigmas`
- Category: `sampling/custom_sampling/sigmas`
- Output node: `False`

The `SplitSigmas` node is designed to partition a sequence of sigma values into two subsets based on a specified step index. This functionality is crucial for custom sampling strategies in generative models, where manipulating the noise levels can significantly impact the model's output quality and diversity.
## Input types
### Required
- **`sigmas`**
    - The sequence of sigma values to be split. This parameter is central to the node's operation as it determines the basis for the partitioning process.
    - Comfy dtype: `SIGMAS`
    - Python dtype: `torch.Tensor`
- **`step`**
    - The index at which the sigma sequence is split into two subsets. This parameter directly influences the composition of the resulting sigma subsets, affecting the sampling process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`high_sigmas`**
    - Comfy dtype: `SIGMAS`
    - The subset of sigma values before the specified step index, representing higher noise levels.
    - Python dtype: `torch.Tensor`
- **`low_sigmas`**
    - Comfy dtype: `SIGMAS`
    - The subset of sigma values from the specified step index onwards, representing lower noise levels.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)



## Source code
```python
class SplitSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, step):
        sigmas1 = sigmas[:step + 1]
        sigmas2 = sigmas[step:]
        return (sigmas1, sigmas2)

```
