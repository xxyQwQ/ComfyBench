---
tags:
- LoRA
---

# Load LoRA
## Documentation
- Class name: `LoraLoader`
- Category: `loaders`
- Output node: `False`

The LoraLoader node is designed to dynamically load and apply LoRA (Low-Rank Adaptation) adjustments to models and CLIP instances based on specified parameters. It facilitates the customization of model behavior and performance without altering the original model architecture, enabling fine-tuned control over model and CLIP instance characteristics through LoRA parameters.
## Input types
### Required
- **`model`**
    - The model to which LoRA adjustments will be applied. It is crucial for defining the base model that will be modified.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The CLIP instance to which LoRA adjustments will be applied. This parameter allows for the modification of CLIP instances alongside models.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`lora_name`**
    - The name of the LoRA file to be loaded. This determines which LoRA adjustments are applied to the model and CLIP instance.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_model`**
    - Defines the strength of LoRA adjustments applied to the model. This parameter allows for fine-tuning the impact of LoRA on the model.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_clip`**
    - Defines the strength of LoRA adjustments applied to the CLIP instance. This parameter allows for adjusting the influence of LoRA on the CLIP instance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with LoRA adjustments applied.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The modified CLIP instance with LoRA adjustments applied.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [LoraLoader](../../Comfy/Nodes/LoraLoader.md)
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - Reroute
    - [VideoLinearCFGGuidance](../../Comfy/Nodes/VideoLinearCFGGuidance.md)
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [FaceDetailer](../../ComfyUI-Impact-Pack/Nodes/FaceDetailer.md)
    - [ModelSamplingDiscrete](../../Comfy/Nodes/ModelSamplingDiscrete.md)
    - [ADE_AnimateDiffLoaderWithContext](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_AnimateDiffLoaderWithContext.md)
    - KSampler //Inspire
    - [ToBasicPipe](../../ComfyUI-Impact-Pack/Nodes/ToBasicPipe.md)



## Source code
```python
class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)

```
