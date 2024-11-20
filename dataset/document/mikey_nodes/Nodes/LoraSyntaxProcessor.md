---
tags:
- Prompt
- Text
---

# Lora Syntax Processor (Mikey)
## Documentation
- Class name: `LoraSyntaxProcessor`
- Category: `Mikey/Lora`
- Output node: `False`

The LoraSyntaxProcessor node is designed to interpret and process specific syntax within text inputs related to LoRA (Low-Rank Adaptation) configurations. It dynamically modifies model and clip parameters based on the LoRA specifications found within the text, enabling customized model behavior and output generation.
## Input types
### Required
- **`model`**
    - The model parameter represents the deep learning model to which LoRA configurations will be applied. It plays a crucial role in the node's operation by being dynamically modified based on the LoRA specifications extracted from the input text.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The clip parameter refers to a specific component or aspect of the model that is targeted for modification through LoRA configurations. It is essential for applying the LoRA adjustments to the correct part of the model.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`text`**
    - The text input contains the specific LoRA syntax that will be parsed and processed to extract LoRA configurations. This syntax directs how the model and clip parameters are to be modified.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`seed`**
    - The seed parameter is used for initializing random number generators, ensuring that any randomness in the LoRA processing is reproducible.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified deep learning model after applying the LoRA configurations extracted from the input text.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The modified clip component of the model after LoRA adjustments have been applied.
    - Python dtype: `torch.nn.Module`
- **`text`**
    - Comfy dtype: `STRING`
    - The input text with all LoRA syntax removed, leaving only the original content for further processing or use.
    - Python dtype: `str`
- **`unprocessed_text`**
    - Comfy dtype: `STRING`
    - The original text input before any LoRA syntax processing, providing a reference to the initial state of the text.
    - Python dtype: `str`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class LoraSyntaxProcessor:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "model": ("MODEL",),
                    "clip": ("CLIP",),
                    "text": ("STRING", {"multiline": True, "default": "<lora:filename:weight>"}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
                    },
                "hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
                }

    RETURN_TYPES = ('MODEL','CLIP','STRING','STRING')
    RETURN_NAMES = ('model','clip','text','unprocessed_text')
    FUNCTION = 'process'
    CATEGORY = 'Mikey/Lora'

    def process(self, model, clip, text, seed, extra_pnginfo=None, prompt=None):
        # process random syntax
        text = process_random_syntax(text, seed)
        # search and replace
        text = search_and_replace(text, extra_pnginfo, prompt)
        lora_re = r'<lora:(.*?)(?::(.*?))?>'
        # find all lora prompts
        lora_prompts = re.findall(lora_re, text)
        stripped_text = text
        # if we found any lora prompts
        clip_lora = clip
        if len(lora_prompts) > 0:
            # loop through each lora prompt
            for lora_prompt in lora_prompts:
                # get the lora filename
                lora_filename = lora_prompt[0]
                # check for file extension in filename
                if '.safetensors' not in lora_filename:
                    lora_filename += '.safetensors'
                # get the lora multiplier
                try:
                    lora_multiplier = float(lora_prompt[1]) if lora_prompt[1] != '' else 1.0
                except:
                    lora_multiplier = 1.0
                model, clip = load_lora(model, clip, lora_filename, lora_multiplier, lora_multiplier)
        # strip lora syntax from text
        stripped_text = re.sub(lora_re, '', stripped_text)
        return (model, clip, stripped_text,  text, )

```
