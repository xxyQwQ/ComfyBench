---
tags:
- Prompt
- PromptComposer
---

# One Button Preset
## Documentation
- Class name: `OneButtonPreset`
- Category: `OneButtonPrompt`
- Output node: `False`

The OneButtonPreset node is designed to streamline the process of applying predefined settings to generate prompts, incorporating elements like insanity level, subject, artist, and various subject subtypes. It simplifies the customization and generation of prompts by utilizing a set of predefined or user-defined presets.
## Input types
### Required
- **`OneButtonPreset`**
    - Specifies the selected preset to apply. This can be a predefined preset or a custom one defined by the user, influencing the overall theme and parameters of the generated prompt.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
### Optional
- **`base_model`**
    - Defines the base model to be used for prompt generation, affecting the style and structure of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`prompt_enhancer`**
    - An optional modifier that applies additional transformations or enhancements to the prompt, further customizing the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`preset_prefix`**
    - A prefix added to the prompt, allowing for further customization and refinement of the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`preset_suffix`**
    - A suffix added to the prompt, used to append additional information or styling to the generated content.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - Determines the randomness seed for generating prompts, ensuring reproducibility of results when the same seed is used.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`prompt`**
    - Comfy dtype: `STRING`
    - The generated prompt based on the applied preset, incorporating any specified customizations such as prefix, suffix, and enhancer.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class OneButtonPreset:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
               
        return {
            "required": {
                "OneButtonPreset": (allpresets, {"default": "Standard"}),
            },
            "optional": {
                "base_model":(models, {"default": "SDXL"}),
                "prompt_enhancer":(prompt_enhancers, {"default": "none"}),
                "preset_prefix": ("STRING", {
                    "multiline": False, # prefix the preset
                    "default": ""
                }),
                "preset_suffix": ("STRING", {
                    "multiline": False, # Suffix of the preset
                    "default": ""
                }),   
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }


    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)

    FUNCTION = "Comfy_OBP_OneButtonPreset"

    #OUTPUT_NODE = False

    CATEGORY = "OneButtonPrompt"
    
    def Comfy_OBP_OneButtonPreset(self, OneButtonPreset, seed, base_model, prompt_enhancer, preset_prefix, preset_suffix):
        # load the stuff
        if(OneButtonPreset == OBPresets.RANDOM_PRESET_OBP):
            selected_opb_preset = OBPresets.get_obp_preset("Standard")
        else:
            selected_opb_preset = OBPresets.get_obp_preset(OneButtonPreset)
        
        insanitylevel=selected_opb_preset["insanitylevel"]
        subject=selected_opb_preset["subject"]
        artist=selected_opb_preset["artist"]
        chosensubjectsubtypeobject=selected_opb_preset["chosensubjectsubtypeobject"]
        chosensubjectsubtypehumanoid=selected_opb_preset["chosensubjectsubtypehumanoid"]
        chosensubjectsubtypeconcept=selected_opb_preset["chosensubjectsubtypeconcept"]
        chosengender=selected_opb_preset["chosengender"]
        imagetype=selected_opb_preset["imagetype"]
        imagemodechance=selected_opb_preset["imagemodechance"]
        givensubject=selected_opb_preset["givensubject"]
        smartsubject=selected_opb_preset["smartsubject"]
        givenoutfit=selected_opb_preset["givenoutfit"]
        prefixprompt=selected_opb_preset["prefixprompt"]
        suffixprompt=selected_opb_preset["suffixprompt"]
        giventypeofimage=selected_opb_preset["giventypeofimage"]
        antistring=selected_opb_preset["antistring"]
        
        generatedprompt = build_dynamic_prompt(insanitylevel=insanitylevel,
                                               forcesubject=subject,
                                               artists=artist,
                                               subtypeobject=chosensubjectsubtypeobject,
                                               subtypehumanoid=chosensubjectsubtypehumanoid,
                                               subtypeconcept=chosensubjectsubtypeconcept,
                                               gender=chosengender,
                                               imagetype=imagetype,
                                               imagemodechance=imagemodechance,
                                               givensubject=givensubject,
                                               smartsubject=smartsubject,
                                               overrideoutfit=givenoutfit,
                                               prefixprompt=prefixprompt,
                                               suffixprompt=suffixprompt,
                                               giventypeofimage=giventypeofimage,
                                               antivalues=antistring,
                                               advancedprompting=False,
                                               hardturnoffemojis=True,
                                               seed=seed,
                                               base_model=base_model,
                                               OBP_preset=OneButtonPreset,
                                               prompt_enhancer=prompt_enhancer,
                                               preset_prefix=preset_prefix,
                                               preset_suffix=preset_suffix,
                                               )
        
        
        return (generatedprompt,)

```
