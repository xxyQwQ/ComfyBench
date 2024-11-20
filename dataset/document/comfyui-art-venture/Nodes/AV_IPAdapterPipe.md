# IP Adapter Pipe
## Documentation
- Class name: `AV_IPAdapterPipe`
- Category: `Art Venture/IP Adapter`
- Output node: `False`

The AV_IPAdapterPipe node is designed to facilitate the integration of IP adapter and clip vision models into a unified pipeline for enhancing image processing tasks. It abstracts the complexity of loading and applying these models, providing a streamlined interface for incorporating advanced vision capabilities into applications.
## Input types
### Required
- **`ip_adapter_name`**
    - Specifies the name of the IP adapter model to be loaded. It is crucial for identifying which IP adapter model to use in the pipeline, affecting the overall image processing outcome.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`clip_name`**
    - Determines the specific clip vision model to be loaded. This selection influences the vision processing capabilities of the pipeline, impacting the final image enhancement results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
## Output types
- **`p`**
    - Comfy dtype: `IPADAPTER`
    - Outputs a pipeline dictionary containing the loaded IP adapter and clip vision models, ready for application in image processing tasks.
    - Python dtype: `Dict[str, Dict[str, Any]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
    class AV_IPAdapterPipe:
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "ip_adapter_name": (folder_paths.get_filename_list("ipadapter"),),
                    "clip_name": (folder_paths.get_filename_list("clip_vision"),),
                }
            }

        RETURN_TYPES = ("IPADAPTER",)
        RETURN_NAMES = "pipeline"
        CATEGORY = "Art Venture/IP Adapter"
        FUNCTION = "load_ip_adapter"

        def load_ip_adapter(self, ip_adapter_name, clip_name):
            ip_adapter = loader.load_ipadapter_model(ip_adapter_name)[0]

            clip_path = folder_paths.get_full_path("clip_vision", clip_name)
            clip_vision = comfy.clip_vision.load(clip_path)

            pipeline = {"ipadapter": {"model": ip_adapter}, "clipvision": {"model": clip_vision}}
            return (pipeline,)

```
