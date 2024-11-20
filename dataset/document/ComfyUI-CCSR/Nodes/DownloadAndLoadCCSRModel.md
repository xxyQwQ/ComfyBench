# DownloadAndLoad CCSRModel
## Documentation
- Class name: `DownloadAndLoadCCSRModel`
- Category: `CCSR`
- Output node: `False`

This node is responsible for downloading and loading the CCSR model into memory, ensuring it is ready for use. It handles the model's retrieval, configuration, and initialization, adapting to the required precision (FP16 or FP32) based on the model's specifications.
## Input types
### Required
- **`model`**
    - Specifies the model variant to be loaded, influencing the precision (FP16 or FP32) and the specific model configuration to be used.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`ccsr_model`**
    - Comfy dtype: `CCSRMODEL`
    - Provides the loaded CCSR model along with its data type, ready for further processing or inference.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class DownloadAndLoadCCSRModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                    [ 
                    'real-world_ccsr-fp16.safetensors',
                    'real-world_ccsr-fp32.safetensors'
                    ],
                ),
            },
        }

    RETURN_TYPES = ("CCSRMODEL",)
    RETURN_NAMES = ("ccsr_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "CCSR"

    def loadmodel(self, model):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        dtype = torch.float16 if 'fp16' in model else torch.float32

        model_path = os.path.join(folder_paths.models_dir, "CCSR")
        safetensors_path = os.path.join(model_path, model)
        
        if not os.path.exists(safetensors_path):
            print(f"Downloading CCSR model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/ccsr-safetensors",
                            allow_patterns=[f'*{model}*'],
                            local_dir=model_path,
                            local_dir_use_symlinks=False)
            
        config_path = os.path.join(script_directory, "configs/model/ccsr_stage2.yaml")
        config = OmegaConf.load(config_path)

        model = instantiate_from_config(config)

        sd = comfy.utils.load_torch_file(safetensors_path)
      
        model.load_state_dict(sd, strict=False)
        del sd
        mm.soft_empty_cache()
        
        ccsr_model = {
            'model': model, 
            'dtype': dtype,
            }

        return (ccsr_model,)

```
