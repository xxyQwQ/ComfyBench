---
tags:
- LoRA
---

# pipeLoraStack
## Documentation
- Class name: `ttN pipeLoraStack`
- Category: `🌏 tinyterra/pipe`
- Output node: `False`

The node is designed to integrate and manage LoRA (Low-Rank Adaptation) models within a processing pipeline, enabling the dynamic adjustment of model and clip strengths based on specified LoRA configurations. It facilitates the customization of AI model behavior for specific tasks or datasets by applying LoRA modifications to the underlying models.
## Input types
### Required
- **`toggle`**
    - A boolean toggle to enable or disable the application of LoRA models within the pipeline.
    - Comfy dtype: `COMBO[BOOLEAN]`
    - Python dtype: `bool`
- **`mode`**
    - Specifies the operation mode, either 'simple' or 'advanced', determining the complexity of LoRA model adjustments.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`num_loras`**
    - The number of LoRA models to be applied, allowing for multiple adaptations.
    - Comfy dtype: `INT`
    - Python dtype: `int`
### Optional
- **`optional_pipe`**
    - An optional pipeline configuration that can be modified by the LoRA adjustments.
    - Comfy dtype: `PIPE_LINE`
    - Python dtype: `dict`
- **`model_override`**
    - Allows for overriding the default model with a specified one for further customization.
    - Comfy dtype: `MODEL`
    - Python dtype: `object`
- **`clip_override`**
    - Enables the override of the default clip model, facilitating custom clip adaptations.
    - Comfy dtype: `CLIP`
    - Python dtype: `object`
- **`optional_lora_stack`**
    - A stack of LoRA models that can be optionally applied for layered adaptations.
    - Comfy dtype: `LORA_STACK`
    - Python dtype: `list`
- **`lora_i_name`**
    - Specifies the name of the ith LoRA model to be applied, determining the specific adaptation for each model.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`lora_i_strength`**
    - Defines the strength of the ith LoRA model adaptation, influencing the degree of modification.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_i_model_strength`**
    - Determines the model-specific strength of the ith LoRA adaptation, affecting the model's behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_i_clip_strength`**
    - Specifies the clip-specific strength of the ith LoRA adaptation, modifying the clip's influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`optional_pipe`**
    - Comfy dtype: `PIPE_LINE`
    - The modified pipeline configuration after applying LoRA adjustments, including any model or clip overrides.
    - Python dtype: `dict`
- **`lora_stack`**
    - Comfy dtype: `LORA_STACK`
    - A stack of LoRA models that have been applied, detailing each model's name and its model and clip strengths.
    - Python dtype: `list`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ttN_pipeLoraStack:
    version = '1.1.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "toggle": ([True, False],),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": 20}),
            },
            "optional": {
                "optional_pipe": ("PIPE_LINE", {"default": None}),
                "model_override": ("MODEL",),
                "clip_override": ("CLIP",),
                "optional_lora_stack": ("LORA_STACK",),
            }, 
            "hidden": {
                "ttNnodeVersion": (ttN_pipeLoraStack.version),
            },
        }

        for i in range(1, 21):
            inputs["optional"][f"lora_{i}_name"] = (["None"] + folder_paths.get_filename_list("loras"),{"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}) 
            inputs["optional"][f"lora_{i}_model_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        
        return inputs
    
    
    RETURN_TYPES = ("PIPE_LINE", "LORA_STACK",)
    RETURN_NAMES = ("optional_pipe","lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "🌏 tinyterra/pipe"

    def stack(self, toggle, mode, num_loras, optional_pipe=None, lora_stack=None, model_override=None, clip_override=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return optional_pipe, None
        
        loras = []

        # Import Stack values
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        
        if not loras:
            return optional_pipe, None
        
        if loras and not optional_pipe:
            return optional_pipe, loras
        
        # Load Loras
        model = model_override or optional_pipe.get("model")
        clip = clip_override or optional_pipe.get("clip")

        if not model or not clip:
            return optional_pipe, loras
        
        for lora in loras:
            model, clip = loader.load_lora(lora[0], model, clip, lora[1], lora[2])

        new_pipe = {
            "model": model,
            "positive": optional_pipe["positive"],
            "negative": optional_pipe["negative"],
            "vae": optional_pipe["vae"],
            "clip": clip,

            "samples": optional_pipe["samples"],
            "images": optional_pipe["images"],
            "seed": optional_pipe["seed"],

            "loader_settings": optional_pipe["loader_settings"],
        }

        del optional_pipe

        return new_pipe, loras

```
