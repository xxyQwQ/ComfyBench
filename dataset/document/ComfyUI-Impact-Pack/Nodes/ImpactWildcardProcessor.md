---
tags:
- Prompt
- Text
- Wildcard
---

# ImpactWildcardProcessor
## Documentation
- Class name: `ImpactWildcardProcessor`
- Category: `ImpactPack/Prompt`
- Output node: `False`

The ImpactWildcardProcessor node is designed to enhance text inputs by dynamically populating them with predefined or custom wildcards. It leverages a comprehensive system for wildcard management, including loading, normalization, and replacement, to inject variability and specificity into text. This node supports both fixed and randomized text generation modes, allowing for controlled or stochastic text transformations based on the provided wildcards.
## Input types
### Required
- **`wildcard_text`**
    - The text containing wildcards that need to be populated or replaced. It serves as the template for generating the final text output.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`populated_text`**
    - The text after wildcards have been populated. This represents the output of the wildcard processing, ready for further use or display.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`mode`**
    - A boolean flag indicating the operation mode: 'Populate' to dynamically replace wildcards in the text, or 'Fixed' to use a predetermined set of replacements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`seed`**
    - An integer seed for random number generation, ensuring reproducibility of the wildcard replacement process when generating text.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`Select to add Wildcard`**
    - Allows selection of specific wildcards to add to the text, providing a way to customize the text generation process further.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
## Output types
- **`string`**
    - Comfy dtype: `STRING`
    - The text after all applicable wildcards have been processed and replaced, reflecting the final output of the node's operation.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)



## Source code
```python
class ImpactWildcardProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        "Select to add Wildcard": (["Select the Wildcard to add to the text"],),
                    },
                }

    CATEGORY = "ImpactPack/Prompt"

    RETURN_TYPES = ("STRING", )
    FUNCTION = "doit"

    @staticmethod
    def process(**kwargs):
        return impact.wildcards.process(**kwargs)

    def doit(self, *args, **kwargs):
        populated_text = ImpactWildcardProcessor.process(text=kwargs['populated_text'], seed=kwargs['seed'])
        return (populated_text, )

```
