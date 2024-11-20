---
tags:
- Checkpoint
---

# CCSR_Model_Select
## Documentation
- Class name: `CCSR_Model_Select`
- Category: `CCSR`
- Output node: `False`

This node is designed to select and load a specific checkpoint for the CCSR model based on the provided checkpoint name. It initializes or updates the CCSR model in the system with the selected checkpoint, ensuring the model is ready for subsequent image processing tasks.
## Input types
### Required
- **`ckpt_name`**
    - The name of the checkpoint to load for the CCSR model. This parameter is crucial for identifying which specific model checkpoint to initialize or update the CCSR model with.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`ccsr_model`**
    - Comfy dtype: `CCSRMODEL`
    - The loaded CCSR model along with its data type, ready for use in image processing tasks. The dictionary contains the model itself and its data type, providing detailed information necessary for image processing applications.
    - Python dtype: `Dict[str, Union[torch.nn.Module, torch.dtype]]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CCSR_Model_Select:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),                                             
                             }}
    RETURN_TYPES = ("CCSRMODEL",)
    RETURN_NAMES = ("ccsr_model",)
    FUNCTION = "load_ccsr_checkpoint"

    CATEGORY = "CCSR"

    def load_ccsr_checkpoint(self, ckpt_name):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")
        dtype = torch.float16 if mm.should_use_fp16() and not mm.is_device_mps(device) else torch.float32
        
        if not hasattr(self, "model") or self.model is None:
            config = OmegaConf.load(config_path)
            self.model = instantiate_from_config(config)
            
            load_state_dict(self.model, comfy.utils.load_torch_file(ckpt_path), strict=True)
            # reload preprocess model if specified

        ccsr_model = {
            'model': self.model, 
            'dtype': dtype
            }
        return (ccsr_model,)

```
