---
tags:
- Prompt
- Text
- Wildcard
---

# ImpactWildcardEncode
## Documentation
- Class name: `ImpactWildcardEncode`
- Category: `ImpactPack/Prompt`
- Output node: `False`

The ImpactWildcardEncode node is designed to dynamically encode text by replacing specified wildcard patterns with corresponding values or options. It leverages a comprehensive wildcard system to interpret and transform text inputs based on predefined or custom wildcard dictionaries, supporting complex pattern matching and replacement strategies to tailor text outputs for varied applications.
## Input types
### Required
- **`model`**
    - This input specifies the model to be used in the encoding process, enabling the node to leverage specific model capabilities for text transformation.
    - Comfy dtype: `MODEL`
    - Python dtype: `str`
- **`clip`**
    - The 'clip' parameter represents the CLIP model used for encoding, providing a mechanism for integrating visual context into the text transformation process.
    - Comfy dtype: `CLIP`
    - Python dtype: `str`
- **`wildcard_text`**
    - This input contains the text with wildcard patterns that the node will interpret and replace, serving as the primary source for the encoding operation.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`populated_text`**
    - The text resulting from the wildcard replacement process, which is further processed or utilized within the node's workflow.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`mode`**
    - A boolean input that toggles between different modes of operation, such as 'Populate' or 'Fixed', affecting how text is processed and wildcards are handled.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`Select to add LoRA`**
    - Allows selection of a LoRA to add to the text, introducing specific logic or rules-based adjustments to the encoding process.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`Select to add Wildcard`**
    - Enables the selection of additional wildcards to be added to the text, expanding the node's capability to transform and customize text outputs.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `list`
- **`seed`**
    - A numerical input that seeds the randomization process, ensuring reproducibility and consistency in the text transformation outcomes.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model output, potentially modified or utilized within the node's processing flow.
    - Python dtype: `str`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP model output, reflecting any changes or usage within the node's operations.
    - Python dtype: `str`
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - Conditioning information generated or modified by the node, used in further processing steps.
    - Python dtype: `str`
- **`populated_text`**
    - Comfy dtype: `STRING`
    - The final text output, with all specified wildcards replaced and processed according to the node's logic and parameters.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - Reroute
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)



## Source code
```python
class ImpactWildcardEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "model": ("MODEL",),
                        "clip": ("CLIP",),
                        "wildcard_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "populated_text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                        "mode": ("BOOLEAN", {"default": True, "label_on": "Populate", "label_off": "Fixed"}),
                        "Select to add LoRA": (["Select the LoRA to add to the text"] + folder_paths.get_filename_list("loras"), ),
                        "Select to add Wildcard": (["Select the Wildcard to add to the text"], ),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    },
                }

    CATEGORY = "ImpactPack/Prompt"

    RETURN_TYPES = ("MODEL", "CLIP", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "clip", "conditioning", "populated_text")
    FUNCTION = "doit"

    @staticmethod
    def process_with_loras(**kwargs):
        return impact.wildcards.process_with_loras(**kwargs)

    @staticmethod
    def get_wildcard_list():
        return impact.wildcards.get_wildcard_list()

    def doit(self, *args, **kwargs):
        populated = kwargs['populated_text']
        processed = []
        model, clip, conditioning = impact.wildcards.process_with_loras(wildcard_opt=populated, model=kwargs['model'], clip=kwargs['clip'], seed=kwargs['seed'], processed=processed)
        return model, clip, conditioning, processed[0]

```
