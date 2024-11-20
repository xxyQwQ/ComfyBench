---
tags:
- LoRA
---

# LoraLoaderModelOnly
## Documentation
- Class name: `LoraLoaderModelOnly`
- Category: `loaders`
- Output node: `False`

The LoraLoaderModelOnly node is designed to load and apply LoRA (Low-Rank Adaptation) modifications to a given model, enhancing its capabilities or adapting it for specific tasks without altering the original model architecture. It focuses on applying these modifications solely to the model, disregarding any other components such as CLIP or additional inputs.
## Input types
### Required
- **`model`**
    - The model to which LoRA modifications will be applied. This parameter is crucial as it determines the base model that will be enhanced with LoRA parameters.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`lora_name`**
    - The name of the LoRA file to be applied to the model. This parameter specifies which LoRA modifications are to be loaded and applied, directly influencing the model's adaptation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_model`**
    - A floating-point value that determines the strength of the LoRA modifications applied to the model. It allows for fine-tuning the impact of LoRA on the model's behavior.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with LoRA adaptations applied. This output showcases the enhanced capabilities or specific task adaptations of the original model through LoRA.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - IPAdapterApplyFaceID
    - [LoraLoaderModelOnly](../../Comfy/Nodes/LoraLoaderModelOnly.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [FreeU_V2](../../Comfy/Nodes/FreeU_V2.md)
    - [UltimateSDUpscale](../../ComfyUI_UltimateSDUpscale/Nodes/UltimateSDUpscale.md)



## Source code
```python
class LoraLoaderModelOnly(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)

```
