---
tags:
- Face
---

# MarigoldModelLoader
## Documentation
- Class name: `MarigoldModelLoader`
- Category: `Marigold`
- Output node: `False`

This node is designed to load and initialize Marigold models for further processing. It prepares the model environment, ensuring that the necessary configurations and resources are in place for efficient execution of depth estimation or other related tasks.
## Input types
### Required
- **`model`**
    - Specifies the model to be loaded. This parameter determines the type of Marigold model that will be initialized for use in subsequent processing steps, affecting the capabilities and performance of the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`marigold_model`**
    - Comfy dtype: `MARIGOLDMODEL`
    - The loaded Marigold model, encapsulated within a dictionary that includes the pipeline and model type. This output is ready for use in depth estimation or other image processing tasks.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class MarigoldModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {            
            "model": (
            ['marigold-v1-0',
             'marigold-lcm-v1-0',
             'marigold-normals-v0-1',
             'marigold-normals-lcm-v0-1',], 
            {
               "default": 'marigold-lcm-v1-0'
            }),
            },
            }
    
    RETURN_TYPES = ("MARIGOLDMODEL",)
    RETURN_NAMES =("marigold_model",)
    FUNCTION = "load"
    CATEGORY = "Marigold"
    DESCRIPTION = """
Diffusion-based monocular depth estimation:  
https://github.com/prs-eth/Marigold  
  
Uses Diffusers 0.28.0 Marigold pipelines.  
Models are automatically downloaded to  
ComfyUI/models/diffusers -folder
"""

    def load(self, model):
        try:
            from diffusers import MarigoldDepthPipeline, MarigoldNormalsPipeline
        except:
            raise Exception("diffusers==0.28 is required for v2 nodes")
        
        device = model_management.get_torch_device()
        diffusers_model_path = os.path.join(folder_paths.models_dir,'diffusers')
        checkpoint_path = os.path.join(diffusers_model_path, model)

        if not os.path.exists(checkpoint_path):
            print(f"Selected model: {checkpoint_path} not found, downloading...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=f"prs-eth/{model}", 
                                allow_patterns=["*.json", "*.txt","*fp16*"],
                                ignore_patterns=["*.bin"],
                                local_dir=checkpoint_path, 
                                local_dir_use_symlinks=False
                                )
        if "normals" in model:
            modeltype = "normals"
            self.marigold_pipeline = MarigoldNormalsPipeline.from_pretrained(
            checkpoint_path, 
            variant="fp16", 
            torch_dtype=torch.float16).to(device)
        else:
            modeltype = "depth"
            self.marigold_pipeline = MarigoldDepthPipeline.from_pretrained(
            checkpoint_path, 
            variant="fp16", 
            torch_dtype=torch.float16).to(device)

        marigold_model = {
            "pipeline": self.marigold_pipeline,
            "modeltype": modeltype
        }
        return (marigold_model,)

```
