---
tags:
- Audio
---

# AudioLDM2 Model Loader
## Documentation
- Class name: `SaltAudioLDM2LoadModel`
- Category: `SALT/AudioViz/Audio/AudioLDM2`
- Output node: `False`

This node is designed to load a specific audio latent diffusion model (AudioLDM2) into memory, making it ready for audio processing tasks. It supports loading different versions of the model and allows specifying the computational device (CPU or GPU) for the model's operations.
## Input types
### Required
- **`model`**
    - Specifies the version of the AudioLDM2 model to load. The choice of model can significantly impact the quality and characteristics of the generated audio.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
### Optional
- **`device`**
    - Determines the computational device ('cuda' for GPU or 'cpu' for CPU) on which the model will be loaded and executed, affecting performance and efficiency.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`audioldm2_model`**
    - Comfy dtype: `AUDIOLDM_MODEL`
    - The loaded AudioLDM2 model, ready for audio processing tasks.
    - Python dtype: `AudioLDM2Pipeline`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class SaltAudioLDM2LoadModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["cvssp/audioldm2", "cvssp/audioldm2-large", "cvssp/audioldm2-music"], ),
            },
            "optional": {
                "device": (["cuda", "cpu"], ),
            },
        }
    
    RETURN_TYPES = ("AUDIOLDM_MODEL", )
    RETURN_NAMES = ("audioldm2_model", )

    FUNCTION = "load_model"
    CATEGORY = f"{MENU_NAME}/{SUB_MENU_NAME}/Audio/AudioLDM2"

    def load_model(self, model, device="cuda"):
        models = folder_paths.models_dir
        audioldm2_models = os.path.join(models, "AudioLDM2")
        return (AudioLDM2Pipeline.from_pretrained(model, cache_dir=audioldm2_models, torch_dtype=torch.float16).to(device), )

```
