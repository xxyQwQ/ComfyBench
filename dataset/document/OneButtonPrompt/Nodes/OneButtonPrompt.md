---
tags:
- Prompt
- PromptComposer
---

# One Button Prompt
## Documentation
- Class name: `OneButtonPrompt`
- Category: `OneButtonPrompt`
- Output node: `False`

The OneButtonPrompt node streamlines the creation and customization of prompts for image generation, offering a suite of options to tailor the prompt's subject, style, and complexity. It leverages a dynamic prompt building mechanism to generate prompts that can vary widely in theme and detail, accommodating a broad range of creative needs.
## Input types
### Required
- **`insanitylevel`**
    - Determines the level of creativity and randomness in the generated prompt, influencing its complexity and uniqueness.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`artist`**
    - Selects artists to influence the style and aesthetic of the generated image, contributing to the prompt's thematic depth.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`imagetype`**
    - Defines the type of image to be generated, such as digital art, painting, or concept art, guiding the visual style of the output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`imagemodechance`**
    - Controls the likelihood of selecting a special image mode for the prompt, adding an element of randomness to the image style.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`subject`**
    - Specifies the main subject for the prompt, offering a base around which the prompt is constructed.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`custom_subject`**
    - Allows for the specification of a custom subject around which the prompt will be built, providing a focused thematic direction.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`custom_outfit`**
    - Specifies an outfit to be included in the prompt, adding detail to the character description.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt_prefix`**
    - Specifies a prefix to be added to the beginning of the prompt, allowing for further customization.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`prompt_suffix`**
    - Specifies a suffix to be added to the end of the prompt, enabling additional tailoring of the prompt's theme.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`humanoids_gender`**
    - Determines the gender of humanoid subjects in the prompt, allowing for more specific character depiction.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`emojis`**
    - Enables or disables the inclusion of emojis in the prompt, affecting its tone and readability.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`base_model`**
    - Chooses the base model for prompt generation, affecting the style and language of the prompt.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`prompt_enhancer`**
    - Selects a prompt enhancer to modify the prompt's style or theme, offering more nuanced control over the generated output.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`seed`**
    - Sets a seed for the random number generator, ensuring reproducibility of prompts when the same inputs are provided.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`prompt`**
    - Comfy dtype: `STRING`
    - The primary generated prompt, ready for use in creative applications.
    - Python dtype: `str`
- **`prompt_g`**
    - Comfy dtype: `STRING`
    - A variant of the generated prompt, offering an alternative perspective or theme.
    - Python dtype: `str`
- **`prompt_l`**
    - Comfy dtype: `STRING`
    - Another variant of the generated prompt, providing additional creative options.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes:
    - [CLIPTextEncode (BlenderNeko Advanced + NSP)](../../was-node-suite-comfyui/Nodes/CLIPTextEncode (BlenderNeko Advanced + NSP).md)
    - [CLIPTextEncodeSDXL](../../Comfy/Nodes/CLIPTextEncodeSDXL.md)
    - ShowText|pysssss
    - [BNK_CLIPTextEncodeAdvanced](../../ComfyUI_ADV_CLIP_emb/Nodes/BNK_CLIPTextEncodeAdvanced.md)
    - DPMagicPrompt
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)



## Source code
```python
class OneButtonPrompt:


    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
               
        return {
            "required": {
                "insanitylevel": ("INT", {
                    "default": 5,
                    "min": 1, #Minimum value
                    "max": 10, #Maximum value
                    "step": 1 #Slider's step
                }),
                },
            "optional": {
                "artist": (artists, {"default": "all"}),
                "imagetype": (imagetypes, {"default": "all"}),
                "imagemodechance": ("INT", {
                    "default": 20,
                    "min": 1, #Minimum value
                    "max": 100, #Maximum value
                    "step": 1 #Slider's step
                }),
                "subject": (subjects, {"default": "------ all"}),
                "custom_subject": ("STRING", {
                    "multiline": False, #True if you want the field to look like the one on the ClipTextEncode node
                    "default": ""
                }),
                "custom_outfit": ("STRING", {
                    "multiline": False, # This is the overwrite for an outfit, super nice
                    "default": ""
                }),
                "prompt_prefix": ("STRING", {
                    "multiline": False, # prefix the prompt
                    "default": ""
                }),
                "prompt_suffix": ("STRING", {
                    "multiline": False, # Suffix of the prompt
                    "default": ""
                }),
                "humanoids_gender": (genders, {"default": "all"}),
                "emojis":(emojis, {"default": False}),
                "base_model":(models, {"default": "SDXL"}),
                "prompt_enhancer":(prompt_enhancers, {"default": "none"}),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING","STRING", "STRING")
    RETURN_NAMES = ("prompt","prompt_g", "prompt_l")

    FUNCTION = "Comfy_OBP"

    #OUTPUT_NODE = False

    CATEGORY = "OneButtonPrompt"
    
    def Comfy_OBP(self, insanitylevel, custom_subject, seed, artist, imagetype, subject, imagemodechance, humanoids_gender, emojis, custom_outfit, base_model, prompt_enhancer, prompt_prefix, prompt_suffix):
        generatedpromptlist = build_dynamic_prompt(insanitylevel,subject,artist,imagetype,False,"",prompt_prefix,prompt_suffix,1,"",custom_subject,True,"",imagemodechance, humanoids_gender,"all", "all", "all", False, emojis, seed, custom_outfit, True, base_model, "", prompt_enhancer)
        #print(generatedprompt)
        generatedprompt = generatedpromptlist[0]
        prompt_g = generatedpromptlist[1]
        prompt_l = generatedpromptlist[2]

        return (generatedprompt, prompt_g, prompt_l)

```
