---
tags:
- RandomGeneration
- Seed
---

# seed
## Documentation
- Class name: `ttN seed`
- Category: `🌏 tinyterra/util`
- Output node: `True`

The ttN seed node is designed to manage and apply seed values for generating or manipulating data within the TinyTerra ecosystem. It integrates with various components such as models, conditioning parameters, and latent spaces to ensure consistent and controlled outcomes. This node plays a crucial role in the procedural generation or transformation of content, allowing for reproducibility and customization through seed manipulation.
## Input types
### Required
- **`seed`**
    - The seed parameter is used to initialize the random number generator, ensuring that the generation or manipulation of data can be reproducible. It is crucial for achieving consistent results across different runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`seed`**
    - Comfy dtype: `INT`
    - The seed value used in the process, ensuring reproducibility and consistency of the results.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CLIPTextEncode (BlenderNeko Advanced + NSP)](../../was-node-suite-comfyui/Nodes/CLIPTextEncode (BlenderNeko Advanced + NSP).md)
    - [OneButtonPrompt](../../OneButtonPrompt/Nodes/OneButtonPrompt.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - KRestartSamplerAdv
    - [DetailerForEachDebug](../../ComfyUI-Impact-Pack/Nodes/DetailerForEachDebug.md)



## Source code
```python
class ttN_SEED:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"ttNnodeVersion": ttN_SEED.version},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "plant"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/util"

    @staticmethod
    def plant(seed):
        return seed,

```
