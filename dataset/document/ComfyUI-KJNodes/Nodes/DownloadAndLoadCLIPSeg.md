# (Down)load CLIPSeg
## Documentation
- Class name: `DownloadAndLoadCLIPSeg`
- Category: `KJNodes/masking`
- Output node: `False`

This node is designed to download and load a specific CLIPSeg model using the huggingface_hub library, making it available for image segmentation tasks within the ComfyUI framework. It abstracts the complexities of downloading, caching, and initializing the model, providing a streamlined way to integrate advanced segmentation capabilities into ComfyUI applications.
## Input types
### Required
- **`model`**
    - Specifies the model to be downloaded and loaded for image segmentation. It supports specific CLIPSeg models available on huggingface_hub, with a default option provided for convenience.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
## Output types
- **`clipseg_model`**
    - Comfy dtype: `CLIPSEGMODEL`
    - Returns a dictionary containing the loaded CLIPSeg model and its processor, ready for performing image segmentation tasks.
    - Python dtype: `Dict[str, Any]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class DownloadAndLoadCLIPSeg:

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
       
        return {"required":
                    {     
                    "model": (
                    [   'Kijai/clipseg-rd64-refined-fp16',
                        'CIDAS/clipseg-rd64-refined',
                    ],
                    {
                    "default": 'clipseg-rd64-refined-fp16'
                    }),
                     },
                }

    CATEGORY = "KJNodes/masking"
    RETURN_TYPES = ("CLIPSEGMODEL",)
    RETURN_NAMES = ("clipseg_model",)
    FUNCTION = "segment_image"
    DESCRIPTION = """
Downloads and loads CLIPSeg model with huggingface_hub,  
to ComfyUI/models/clip_seg
"""

    def segment_image(self, model):
        from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        checkpoint_path = os.path.join(folder_paths.models_dir,'clip_seg', model)
        if not hasattr(self, "model"):
            if not os.path.exists(checkpoint_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=model, local_dir=checkpoint_path.split("/")[-1], local_dir_use_symlinks=False)
            self.model = CLIPSegForImageSegmentation.from_pretrained(checkpoint_path)

        processor = CLIPSegProcessor.from_pretrained(checkpoint_path)

        clipseg_model = {}
        clipseg_model['model'] = self.model
        clipseg_model['processor'] = processor

        return clipseg_model,

```
