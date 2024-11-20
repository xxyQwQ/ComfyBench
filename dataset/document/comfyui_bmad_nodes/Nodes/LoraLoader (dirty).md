---
tags:
- LoRA
---

# LoraLoader (dirty)
## Documentation
- Class name: `LoraLoader (dirty)`
- Category: `Bmad/api/dirty loaders`
- Output node: `False`

The LoraLoader (dirty) node is designed to dynamically load LoRA (Low-Rank Adaptation) parameters into models and CLIP instances, allowing for the customization and fine-tuning of these components based on specified LoRA files and strength parameters. It serves as a bridge between static model states and adaptable, performance-enhanced versions tailored to specific tasks or datasets.
## Input types
### Required
- **`model`**
    - The model parameter represents the pre-trained model into which LoRA parameters will be loaded. It is crucial for defining the base architecture that will be enhanced with LoRA adaptations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The clip parameter signifies the CLIP model instance that will be adjusted using LoRA parameters. It plays a key role in adapting the CLIP model for enhanced performance in specific tasks.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`lora_name`**
    - The lora_name parameter specifies the filename of the LoRA parameters to be loaded. It determines which specific LoRA adaptations will be applied to the model and CLIP instances.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`strength_model`**
    - This parameter controls the intensity of the LoRA adaptation applied to the model. It allows for fine-tuning the impact of LoRA parameters on the model's performance.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`strength_clip`**
    - Similar to strength_model, this parameter adjusts the intensity of the LoRA adaptation on the CLIP model, enabling precise control over its influence.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The modified model with LoRA parameters applied, reflecting the adaptations made for enhanced performance or customization.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The adjusted CLIP model with LoRA parameters incorporated, showcasing the enhancements made for improved task-specific performance.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DirtyLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",),
                             "clip": ("CLIP",),
                             "lora_name": ("STRING", {"default": ""}),
                             "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                             }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    CATEGORY = f"{api_category_path}/dirty loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        lora_name = DirtyLoaderUtils.find_matching_filename(
            lora_name, folder_paths.get_filename_list("loras"))

        loader = nodes.LoraLoader()
        return loader.load_lora(model, clip, lora_name, strength_model, strength_clip)

```
